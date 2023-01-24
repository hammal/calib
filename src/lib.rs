pub mod cli;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::{fs::File, path::PathBuf};

pub struct LMSFilter {
    // The filter coefficients in a
    // L x N x K array
    pub h: Array3<f64>,

    // Number of input channels
    pub l: usize,
    // Number of control signals
    pub m: usize,
    // Number of filter taps
    pub k: usize,

    // A gradient vector
    _gradient_step: Array2<f64>,

    _accumulated_error: f64,
}

impl LMSFilter {
    // instantiate a new filter from a numpy array file
    pub fn new(path: &PathBuf) -> Self {
        let filter_file = File::open(path).unwrap();
        let numpy_array = Array3::<f64>::read_npy(filter_file).unwrap();
        let l = numpy_array.shape()[0];
        let m = numpy_array.shape()[1];
        let k = numpy_array.shape()[2];
        Self {
            _gradient_step: Array2::<f64>::zeros((m, k)),
            h: numpy_array,
            l,
            m,
            k,
            _accumulated_error: 0.0,
        }
    }

    pub fn u_hat(&self, control_signal: &ArrayView2<f64>) -> f64 {
        // println!("control signal: {:?}", control_signal.shape());
        // println!("h: {:?}", self.h.shape());
        (&self.h * control_signal).sum()
    }

    pub fn lms_single_update(&mut self, control_signal: &ArrayView2<f64>, step_size: f64) -> f64 {
        let error = self.u_hat(control_signal);
        let step_size_error = step_size * error;
        self._gradient_step
            .slice_mut(s![1..self.m, ..])
            .zip_mut_with(&control_signal.slice(s![1..self.m, ..]), |g, cs| {
                *g = step_size_error * cs;
            });
        self.h -= &self._gradient_step;
        error
    }

    pub fn lms_batch_update(
        &mut self,
        control_signal: &ArrayView2<f64>,
        step_size: f64,
        batch_update: bool,
    ) -> f64 {
        let error = self.u_hat(control_signal);
        let step_size_error = step_size * error;
        self._gradient_step
            .slice_mut(s![1..self.m, ..])
            .zip_mut_with(&control_signal.slice(s![1..self.m, ..]), |g, cs| {
                *g += step_size_error * cs;
            });

        if batch_update {
            self.h -= &self._gradient_step;
            self._gradient_step = Array2::<f64>::zeros(self._gradient_step.dim());
        }
        error
    }

    pub fn save_output(&self, path: &PathBuf) {
        let mut filter_file = File::create(path).unwrap();
        self.h.write_npy(&mut filter_file).unwrap();
    }
}

#[test]
fn test_filter_new() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"));
    assert_eq!(filter.h.shape(), &[1, 7, 512]);
}

#[test]
fn test_filter_save() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"));
    filter.save_output(&PathBuf::from("test_data/filter_save.npy"));
    let filter_save = LMSFilter::new(&PathBuf::from("test_data/filter_save.npy"));
    assert_eq!(filter.h, filter_save.h);
}

#[test]
fn test_filter_u_hat() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"));
    let control_signal = Array2::<f64>::ones((7, 512));

    for i in 0..7 {
        for j in 0..512 {
            println!("{} ", filter.h[[0, i, j]]);
        }
    }
    assert_eq!(
        filter.u_hat(&control_signal.slice(s![.., ..])),
        -0.0988389426867814
    );
}

pub struct ControlSignal {
    pub signal: Array2<f64>,
    pub size: usize,
    k: usize,
    m: usize,
}

impl ControlSignal {
    pub fn new(path: &PathBuf, k: usize) -> Self {
        let temp = Array2::<f64>::read_npy(File::open(path).unwrap()).unwrap();
        let mut control_signal_vector = Array2::<f64>::zeros(temp.t().dim());
        // transpose the control signal vector
        control_signal_vector.assign(&temp.t());
        // apply 2 s - 1
        control_signal_vector.par_map_inplace(|x| {
            *x = 2.0 * (*x) - 1.0;
        });
        let size = control_signal_vector.shape()[1] - k;
        let m: usize = control_signal_vector.shape()[0];
        // println!("control_signal_vector: {:?}", control_signal_vector.shape());
        Self {
            signal: control_signal_vector,
            size: size,
            k: k,
            m: m,
        }
    }

    pub fn next(&self, index: usize) -> ArrayView2<f64> {
        let cyclic_index = index % self.size;
        let signal = self
            .signal
            .slice(s![.., cyclic_index..(cyclic_index + self.k)]);
        signal
    }
}

pub trait Estimator {
    fn new(input_path: &PathBuf, filter_path: &PathBuf) -> Self;
    fn calibrate(&mut self, iterations: usize, step_size: f64, batch_size: usize);
    fn validate(&mut self);
    fn save_output(&self, output: &PathBuf);
    fn save_filter(&self, filter_path: &PathBuf);
}

pub struct LMSEstimator {
    control_signal: ControlSignal,
    filter: LMSFilter,
    result: Array1<f64>,
}

impl LMSEstimator {
    fn _get_progress_bar(&self, size: u64) -> ProgressBar {
        let bar = ProgressBar::new(size);
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed}/{duration}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        bar
    }
}

impl Estimator for LMSEstimator {
    fn new(input_path: &PathBuf, filter_path: &PathBuf) -> Self {
        let filter = LMSFilter::new(filter_path);
        let control_signal = ControlSignal::new(input_path, filter.k);
        let result = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
    }

    fn calibrate(&mut self, iterations: usize, step_size: f64, batch_size: usize) {
        let bar = self._get_progress_bar(iterations as u64);
        let mut index = 0;
        let mut batch_update: bool;
        let temp_div = match iterations / self.control_signal.size {
            0 => {
                1
            }
            _ => {
                iterations / self.control_signal.size
            }
        };
        for i in 0..iterations {
            batch_update = (i % batch_size) == 0 && index != 0;
            index = i * self.control_signal.size / iterations;
            // if (i % temp_div) == 0 && i > 0 {
            //     index += 1;
            // }
            self.result[index] +=
                self.filter
                    .lms_batch_update(&self.control_signal.next(i), step_size, batch_update)
                    / temp_div as f64;
            bar.inc(1);        
        }
        bar.finish();
    }

    fn validate(&mut self) {
        let bar = self._get_progress_bar(self.control_signal.size as u64);
        for index in 0..self.control_signal.size {
            bar.inc(1);
            self.result[index] = self.filter.u_hat(&self.control_signal.next(index));
        }
        bar.finish();
    }

    fn save_output(&self, output: &PathBuf) {
        let mut output_file = File::create(output).unwrap();
        self.result.write_npy(&mut output_file).unwrap();
    }

    fn save_filter(&self, filter_path: &PathBuf) {
        // save the filter
        self.filter.save_output(filter_path);
    }
}


#[test]
fn test_load_control_signals() {
    let control_signal_vector =
        Array2::<f64>::read_npy(File::open("test_data/control_sequence.npy").unwrap()).unwrap();
    assert_eq!(control_signal_vector.shape(), &[16384, 7]);
}

#[test]
fn test_validate() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"));
    let controls = ControlSignal::new(&PathBuf::from("test_data/control_sequence.npy"), filter.k);
    let mut result = Array1::<f64>::zeros(controls.size);
    for index in 0..controls.size {
        result[index] = filter.u_hat(&controls.next(index));
        println!("{}", result[index])
    }
    // let expected_result =
    //     Array1::<f64>::read_npy(File::open("test_data/validate_result.npy").unwrap()).unwrap();
    // assert_eq!(result, expected_result);
}

#[test]
fn test_calibrate() {
    let epochs = 1 << 14;
    let mut filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"));
    let controls = ControlSignal::new(&PathBuf::from("test_data/s_train.npy"), filter.k);
    let mut result = Array1::<f64>::zeros(epochs);
    for index in 0..epochs {
        result[index] = filter.lms_single_update(&controls.next(index), 1e-6);
        println!("{}", result[index])
    }
    // let expected_result =
    //     Array1::<f64>::read_npy(File::open("test_data/validate_result.npy").unwrap()).unwrap();
    // assert_eq!(result, expected_result);
}
