use calib::cli::{get_cli, Commands};
use calib::{Estimator, LMSEstimator};
use env_logger::Env;
use human_panic::setup_panic;
use log::info;

extern crate log;

extern crate exitcode;

fn main() {
    // initialize human panic
    setup_panic!();

    let env = Env::default()
        .filter_or("CALIB_LOG_LEVEL", "info")
        .write_style_or("CALIB_LOG_STYLE", "always");

    // initialize logger
    env_logger::init_from_env(env);

    match get_cli().command {
        Commands::Validate {
            filter,
            input,
            output,
            non_symmetric_dac,
            time_symmetric,
        } => {
            info!("validate was called!");
            let load_offset = true;

            let mut estimator = match (non_symmetric_dac, time_symmetric) {
                (true, false) => {
                    info!("using non-symmetric dac waveform");
                    LMSEstimator::new_non_symmteric_validate(&input, &filter, load_offset)
                }
                (false, false) => {
                    info!("using symmetric dac waveform");
                    LMSEstimator::new(&input, &filter, load_offset)
                }
                (false, true) => {
                    info!("using non-time symmetric dac waveform");
                    LMSEstimator::new_non_time_symmetric_validate(&input, &filter, load_offset)
                }
                (true, true) => {
                    panic!("not implemented")
                }
            };
            estimator.validate();
            estimator.save_output(&output);
        }
        Commands::Calibrate {
            filter,
            input,
            output,
            iterations,
            batch_size,
            step_size,
            decay,
            non_symmetric_dac,
            time_symmetric,
        } => {
            info!("calibrate was called! with {} iterations", iterations);
            if decay > 1.0 || decay < 0.0 {
                panic!("decay must be in [0,1)");
            }

            let load_offset = false;

            let estimator = match (non_symmetric_dac, time_symmetric) {
                (true, false) => {
                    info!("using non-symmetric dac waveform");
                    let mut estimator =
                        LMSEstimator::new_non_symmetric_calibrate(&input, &filter, load_offset);
                    estimator.calibrate(iterations, step_size, decay, batch_size, 2);
                    estimator
                }
                (false, false) => {
                    info!("using symmetric dac waveform");
                    let mut estimator = LMSEstimator::new(&input, &filter, load_offset);
                    estimator.calibrate(iterations, step_size, decay, batch_size, 1);
                    estimator
                }
                (false, true) => {
                    info!("using non-time symmetric dac waveform");
                    let mut estimator = LMSEstimator::new_non_time_symmetric_calibrate(
                        &input,
                        &filter,
                        load_offset,
                    );
                    estimator.calibrate(iterations, step_size, decay, batch_size, 2);
                    estimator
                }
                (true, true) => {
                    panic!("not implemented")
                }
            };
            estimator.save_filter(&filter);
            estimator.save_output(&output);
        }
    }

    std::process::exit(exitcode::OK);
}
