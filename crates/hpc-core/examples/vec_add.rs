use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
    types::CL_BLOCKING,
};

#[cfg(feature = "metrics")]
use hpc_core::dump;


fn main() -> Result<(), ClError> {
    // ----- 1. Context & Queue ----------------------
    let platform   = get_platforms()?.remove(0);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
    let device     = Device::new(device_ids[0]);
    let context    = Context::from_device(&device)?;
    let queue      = CommandQueue::create(&context, device.id(), 0)?;

    // ----- 2. Host‑Daten vorbereiten ---------------
    let n     = 1 << 20;
    let h_a   = vec![1.0_f32; n];
    let h_b   = vec![2.0_f32; n];
    let mut h_out = vec![0.0_f32; n];

    // ----- 3. Device‑Buffer anlegen ----------------
    let a_queued = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;
    let b_queued = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;
    let out_dev  = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;

    // Host → Device kopieren mit Events
    let (a_inflight, guard_a) = a_queued.enqueue_write(&queue, cast_slice(&h_a))?;
    let (b_inflight, guard_b) = b_queued.enqueue_write(&queue, cast_slice(&h_b))?;
    let a_ready = a_inflight.into_ready(guard_a);
    let b_ready = b_inflight.into_ready(guard_b);

    // ----- 4. Kernel kompilieren & argumentieren ----
    let src     = include_str!("vec_add.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "vec_add")?;
    kernel.set_arg(0, a_ready.raw())?;
    kernel.set_arg(1, b_ready.raw())?;
    kernel.set_arg(2, out_dev.raw())?;

    // Kernel ausführen
    let global_size = [n, 1, 1];
    let evt_kernel = queue.enqueue_nd_range_kernel(
        kernel.get(),
        1,
        std::ptr::null(),
        global_size.as_ptr(),
        std::ptr::null(),
        &[],
    )?;

    // Buffer in Ready überführen
    let out_inflight = out_dev.launch();
    let out_ready    = out_inflight.complete(evt_kernel);

    // ----- 5. Device → Host kopieren ----------------
    queue.enqueue_read_buffer(
        out_ready.raw(),
        CL_BLOCKING,
        0,
        cast_slice_mut(&mut h_out),
        &[],
    )?;
    queue.finish()?;

    // ----- 6. Verifizieren --------------------------
    assert!(h_out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    println!("vec_add OK, first element = {}", h_out[0]);


    #[cfg(feature = "metrics")]
     dump();
    

    Ok(())
}