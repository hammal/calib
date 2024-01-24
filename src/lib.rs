pub mod cli;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

const OFFSET_FILENAME: &str = "offset";

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

    // The offset of the filter
    pub offset: f64,

    // A gradient vector
    _gradient_step: Array2<f64>,

    _gradient_offset_step: f64,

    _accumulated_error: f64,
}

impl LMSFilter {
    fn get_offset(offset: bool) -> f64 {
        match offset {
            true => match File::open(PathBuf::from(OFFSET_FILENAME)) {
                Ok(mut file) => {
                    let mut offset_bytes = [0; 8];
                    file.read_exact(&mut offset_bytes).unwrap();
                    let offset = f64::from_be_bytes(offset_bytes);
                    if offset.is_nan() {
                        0.0
                    } else {
                        offset
                    }
                }
                Err(_) => 0.0,
            },
            false => 0.0,
        }
    }

    // instantiate a new filter from a numpy array file
    pub fn new(path: &PathBuf, offset: bool) -> Self {
        let filter_file = File::open(path).unwrap();
        let numpy_array = Array3::<f64>::read_npy(filter_file).unwrap();

        let offset = Self::get_offset(offset);
        println!("offset: {}", offset);

        let l = numpy_array.shape()[0];
        let m = numpy_array.shape()[1];
        let k = numpy_array.shape()[2];
        Self {
            _gradient_step: Array2::<f64>::zeros((m, k)),
            h: numpy_array,
            l,
            m,
            k,
            offset,
            _accumulated_error: 0.0,
            _gradient_offset_step: 0.0,
        }
    }

    pub fn new_non_symmetric(path: &PathBuf, offset: bool) -> Self {
        let filter_file = File::open(path).unwrap();
        let numpy_array = Array3::<f64>::read_npy(filter_file).unwrap();

        let (l, mut m, k) = numpy_array.dim();
        m <<= 1;

        let offset = Self::get_offset(offset);
        println!("offset: {}", offset);

        let mut temp_array = Array3::<f64>::zeros((l, m, k));
        for mm in 0..m / 2 {
            temp_array
                .slice_mut(s![.., 2 * mm, ..])
                .scaled_add(1.0, &numpy_array.slice(s![.., mm, ..]));
            temp_array
                .slice_mut(s![.., 2 * mm + 1, ..])
                .scaled_add(-1.0, &numpy_array.slice(s![.., mm, ..]));
        }

        Self {
            _gradient_step: Array2::<f64>::zeros((m, k)),
            h: temp_array,
            l,
            m,
            k,
            offset,
            _accumulated_error: 0.0,
            _gradient_offset_step: 0.0,
        }
    }

    pub fn new_non_time_symmetric(path: &PathBuf, offset: bool) -> Self {
        let filter_file = File::open(path).unwrap();
        let numpy_array = Array3::<f64>::read_npy(filter_file).unwrap();

        let (l, mut m, k) = numpy_array.dim();
        m <<= 1;

        let offset = Self::get_offset(offset);
        println!("offset: {}", offset);

        let mut temp_array = Array3::<f64>::zeros((l, m, k));
        for mm in 0..m / 2 {
            temp_array
                .slice_mut(s![.., 2 * mm, ..])
                .scaled_add(1.0, &numpy_array.slice(s![.., mm, ..]));
            temp_array
                .slice_mut(s![.., 2 * mm + 1, ..])
                .scaled_add(1.0, &numpy_array.slice(s![.., mm, ..]));
        }

        Self {
            _gradient_step: Array2::<f64>::zeros((m, k)),
            h: temp_array,
            l,
            m,
            k,
            offset,
            _accumulated_error: 0.0,
            _gradient_offset_step: 0.0,
        }
    }

    pub fn u_hat(&self, control_signal: &ArrayView2<f64>) -> f64 {
        // println!("control signal: {:?}", control_signal.shape());
        // println!("h: {:?}", self.h.shape());
        let estimate = (&self.h * control_signal).sum() + self.offset;
        // (&self.h * control_signal).sum_axis(Axis(2)).sum_axis(axis) + self.offset
        if estimate.is_nan() {
            panic!("estimate is nan");
        }
        estimate
    }

    pub fn lms_single_update(
        &mut self,
        control_signal: &ArrayView2<f64>,
        step_size: f64,
        number_of_refs: usize,
    ) -> f64 {
        let error = self.u_hat(control_signal);
        let step_size_error = step_size * error;
        self._gradient_step
            .slice_mut(s![number_of_refs..self.m, ..])
            .zip_mut_with(
                &control_signal.slice(s![number_of_refs..self.m, ..]),
                |g, cs| {
                    *g = step_size_error * cs;
                },
            );
        self.h -= &self._gradient_step;
        self.offset -= step_size_error * error;
        error
    }

    pub fn lms_batch_update(
        &mut self,
        control_signal: &ArrayView2<f64>,
        step_size: f64,
        decay: f64,
        batch_update: bool,
        number_of_refs: usize,
    ) -> f64 {
        let error = self.u_hat(control_signal);
        let step_size_error = step_size * error;
        self._gradient_step
            .slice_mut(s![number_of_refs..self.m, ..])
            .zip_mut_with(
                &control_signal.slice(s![number_of_refs..self.m, ..]),
                |g, cs| {
                    *g += step_size_error * cs;
                },
            );
        self._gradient_offset_step += step_size_error;

        if batch_update {
            self.h -= &self._gradient_step;
            // self._gradient_step = Array2::<f64>::zeros(self._gradient_step.dim());
            self._gradient_step *= decay;
            self.offset -= self._gradient_offset_step;
            self._gradient_offset_step *= decay;
        }
        error
    }

    pub fn save_output(&self, path: &PathBuf) {
        let mut filter_file = File::create(path).unwrap();
        self.h.write_npy(&mut filter_file).unwrap();
        let mut filter_file = File::create(PathBuf::from(OFFSET_FILENAME)).unwrap();
        let temp = self.offset.to_be_bytes();
        filter_file.write_all(&temp).unwrap();
    }
}

#[test]
fn test_filter_new() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"), false);
    assert_eq!(filter.h.shape(), &[1, 7, 512]);
}

#[test]
fn test_filter_save() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"), false);
    filter.save_output(&PathBuf::from("test_data/filter_save.npy"));
    let filter_save = LMSFilter::new(&PathBuf::from("test_data/filter_save.npy"), false);
    assert_eq!(filter.h, filter_save.h);
}

#[test]
fn test_filter_u_hat() {
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"), false);
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

    pub fn new_non_symmetric_controls(path: &PathBuf, k: usize) -> Self {
        let temp = Array2::<f64>::read_npy(File::open(path).unwrap()).unwrap();
        let (size, m) = temp.dim();

        let mut control_signal_vector = Array2::<f64>::zeros((2 * m, size));
        for mm in 0..m {
            control_signal_vector
                .slice_mut(s![2 * mm, ..])
                .iter_mut()
                .zip(temp.slice(s![.., mm]).iter())
                .for_each(|(x, y)| {
                    if *y > 0.5 {
                        *x = 1.0;
                    } else {
                        *x = 0.0;
                    }
                });
            control_signal_vector
                .slice_mut(s![2 * mm + 1, ..])
                .iter_mut()
                .zip(temp.slice(s![.., mm]).iter())
                .for_each(|(x, y)| {
                    if *y < 0.5 {
                        *x = 1.0;
                    } else {
                        *x = 0.0;
                    }
                });
        }
        // println!("original: {:?}", temp.slice(s![.., ..]));
        // println!("control_signal_vector: {:?}", control_signal_vector);
        // println!("control_signal_vector: {:?}", control_signal_vector.shape());
        Self {
            signal: control_signal_vector,
            size: size - k,
            k,
            m: 2 * m,
        }
    }

    pub fn new_non_time_symmetric_controls(path: &PathBuf, k: usize) -> Self {
        let temp = Array2::<f64>::read_npy(File::open(path).unwrap()).unwrap();
        let (size, m) = temp.dim();

        let mut control_signal_vector = Array2::<f64>::zeros((2 * m, size));
        for mm in 0..m {
            control_signal_vector
                .slice_mut(s![2 * mm, ..])
                .iter_mut()
                .zip(temp.slice(s![.., mm]).iter())
                .zip(temp.slice(s![.., mm]).iter().skip(1))
                .for_each(|((x, s_new), s_old)| {
                    if s_new == s_old {
                        *x = 2.0 * s_new - 1.0;
                    }
                });
            control_signal_vector
                .slice_mut(s![2 * mm + 1, ..])
                .iter_mut()
                .zip(temp.slice(s![.., mm]).iter())
                .zip(temp.slice(s![.., mm]).iter().skip(1))
                .for_each(|((x, s_new), s_old)| {
                    if s_new != s_old {
                        *x = 2.0 * s_new - 1.0;
                    }
                });
        }
        // println!("original: {:?}", temp.slice(s![.., ..]));
        // println!("control_signal_vector: {:?}", control_signal_vector);
        // println!("control_signal_vector: {:?}", control_signal_vector.shape());
        Self {
            signal: control_signal_vector,
            size: size - k,
            k,
            m: 2 * m,
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
    fn new(input_path: &PathBuf, filter_path: &PathBuf, load_offset: bool) -> Self;
    fn new_non_symmetric_calibrate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self;
    fn new_non_symmteric_validate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self;
    fn calibrate(
        &mut self,
        iterations: usize,
        step_size: f64,
        decay: f64,
        batch_size: usize,
        number_of_refs: usize,
    );
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
    fn new(input_path: &PathBuf, filter_path: &PathBuf, load_offset: bool) -> Self {
        let filter = LMSFilter::new(filter_path, load_offset);
        let control_signal = ControlSignal::new(input_path, filter.k);

        let result: Array1<f64> = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            println!("control_signal: {:?}", control_signal.m);
            println!("filter: {:?}", filter.m);
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
    }

    fn new_non_symmetric_calibrate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self {
        let filter = LMSFilter::new_non_symmetric(filter_path, load_offset);
        let control_signal = ControlSignal::new_non_symmetric_controls(input_path, filter.k);

        let result: Array1<f64> = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            println!("control_signal: {:?}", control_signal.m);
            println!("filter: {:?}", filter.m);
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
    }
    fn new_non_symmteric_validate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self {
        let filter = LMSFilter::new(filter_path, load_offset);
        let control_signal = ControlSignal::new_non_symmetric_controls(input_path, filter.k);

        let result: Array1<f64> = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            println!("control_signal: {:?}", control_signal.m);
            println!("filter: {:?}", filter.m);
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
    }

    fn calibrate(
        &mut self,
        iterations: usize,
        step_size: f64,
        decay: f64,
        batch_size: usize,
        number_of_refs: usize,
    ) {
        let bar = self._get_progress_bar(iterations as u64);
        let mut index = 0;
        let mut batch_update: bool;
        let temp_div = match iterations / self.control_signal.size {
            0 => 1,
            _ => iterations / self.control_signal.size,
        };
        for i in 0..iterations {
            batch_update = (i % batch_size) == 0 && index != 0;
            index = i * self.control_signal.size / iterations;
            // if (i % temp_div) == 0 && i > 0 {
            //     index += 1;
            // }
            self.result[index] += self.filter.lms_batch_update(
                &self.control_signal.next(i),
                step_size,
                decay,
                batch_update,
                number_of_refs,
            ) / temp_div as f64;
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

impl LMSEstimator {
    pub fn new_non_time_symmetric_calibrate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self {
        let filter = LMSFilter::new_non_time_symmetric(filter_path, load_offset);
        let control_signal = ControlSignal::new_non_time_symmetric_controls(input_path, filter.k);

        let result: Array1<f64> = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            println!("control_signal: {:?}", control_signal.m);
            println!("filter: {:?}", filter.m);
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
    }

    pub fn new_non_time_symmetric_validate(
        input_path: &PathBuf,
        filter_path: &PathBuf,
        load_offset: bool,
    ) -> Self {
        let filter = LMSFilter::new(filter_path, load_offset);
        let control_signal = ControlSignal::new_non_time_symmetric_controls(input_path, filter.k);

        let result: Array1<f64> = Array1::<f64>::zeros(control_signal.size);

        if control_signal.m != filter.m {
            println!("control_signal: {:?}", control_signal.m);
            println!("filter: {:?}", filter.m);
            panic!("Control signal and filter have different number of channels");
        }

        Self {
            control_signal,
            filter,
            result,
        }
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
    let filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"), false);
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
    let mut filter = LMSFilter::new(&PathBuf::from("test_data/filter.npy"), false);
    let controls = ControlSignal::new(&PathBuf::from("test_data/s_train.npy"), filter.k);
    let mut result = Array1::<f64>::zeros(epochs);
    for index in 0..epochs {
        result[index] = filter.lms_single_update(&controls.next(index), 1e-6, 1);
        println!("{}", result[index])
    }
    // let expected_result =
    //     Array1::<f64>::read_npy(File::open("test_data/validate_result.npy").unwrap()).unwrap();
    // assert_eq!(result, expected_result);
}
