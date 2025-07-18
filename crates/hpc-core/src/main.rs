

use opencl3::{context::Context, command_queue::CommandQueue, program::Program};
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel;
use std::{fs, ptr};
use hpc_core::{GpuBuffer, Queued, ClError};

fn main() -> Result<(), hpc_core::ClError> {
    // 1. Gerät & Kontext
    let platform = opencl3::platform::get_platforms()?[0];
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU)?; // <- ARGUMEN
    let device = Device::new(device_ids[0]);
    let context = Context::from_device(&device)?;

    // 2. Kernel einlesen & bauen
    let src = fs::read_to_string("examples/vec_add.cl")
    .map_err(|_| ClError::Api(-2))?;
    let program = Program::create_and_build_from_source(&context, &src, "")
    .map_err(|e| ClError::Api(-3))?;
    let kernel = Kernel::create(&program, "vec_add")
    .map_err(|_| ClError::Api(-4))?;

    // 3. Buffers anlegen (z.B. 1024 floats)
    let n = 1024;
    let buf_a = GpuBuffer::<Queued>::new(&context, n * 4)?;
    let buf_b = GpuBuffer::<Queued>::new(&context, n * 4)?;
    let buf_c = GpuBuffer::<Queued>::new(&context, n * 4)?;

    // TODO: Daten füllen, Kernel argumentieren und ausführen...

    println!("vec_add example setup OK");
    Ok(())
}
