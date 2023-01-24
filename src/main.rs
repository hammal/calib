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
        } => {
            info!("validate was called!");
            let mut estimator = LMSEstimator::new(&input, &filter);
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
        } => {
            info!("calibrate was called! with {} iterations", iterations);
            let mut estimator = LMSEstimator::new(&input, &filter);
            estimator.calibrate(iterations, step_size, batch_size);
            estimator.save_filter(&filter);
            estimator.save_output(&output);
        }
    }

    std::process::exit(exitcode::OK);
}
