use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued};
use opencl3::{command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::{Device, CL_DEVICE_TYPE_GPU}, kernel::Kernel, platform::get_platforms, program::Program, types::CL_BLOCKING};

#[cfg(feature = "metrics")]
use hpc_core::summary;
#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};

fn main() -> Result<(), ClError> {
    /* ---------- OpenCL Setup ---------- */
    let platform   = get_platforms()?.remove(0);
    let dev_id     = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device     = Device::new(dev_id);
    let context    = Context::from_device(&device)?;
    let queue      = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    /* ---------- Problemgröße ---------- */
    const NX: usize = 1024;
    const NY: usize = 1024;
    const N_ITERS: usize = 10;
    const N_BYTES: usize = NX * NY * std::mem::size_of::<f32>();

    /* ---------- Ping‑Pong Buffers ---------- */
    // Ping: initialer Host‑Upload
    let ping_buf = vec![1.0_f32; NX * NY];
    let (ping_if, guard) = GpuBuffer::<Queued>::new(&context, N_BYTES)?
        .enqueue_write(&queue, cast_slice(&ping_buf))?;
    let mut ping = ping_if.into_ready(guard);    // GpuBuffer<Ready>

    /* ---------- Kernel Preparation ---------- */
    let src     = include_str!("stencil.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "jacobi")?;
    kernel.set_arg(2, &(NX as i32))?;  // width
    let global = [NX, NY, 1];

    /* ---------- Jacobi Iterations ---------- */
    for _ in 0..N_ITERS {
        // neues Ziel-Buffer in Queued-State
        let dst = GpuBuffer::<Queued>::new(&context, N_BYTES)?;
        // DST -> InFlight
        let mut dst_if = dst.launch();

        // MemTrace token für Kernel
        #[cfg(feature = "memtrace")]
        let ktok = Box::new(trace_start(Dir::Kernel, 0));

        // Kernel-Argumente
        kernel.set_arg(0, ping.raw())?;
        kernel.set_arg(1, dst_if.raw_mut())?;

        // Kernel-Launch
        let evt = queue.enqueue_nd_range_kernel(
            kernel.get(), 2,
            std::ptr::null(),
            global.as_ptr(),
            std::ptr::null(),
            &[],
        )?;

        // Callback registrieren
        #[cfg(feature = "memtrace")]
        {
            use opencl3::event::CL_COMPLETE;
            let ptr = Box::into_raw(ktok) as *mut std::ffi::c_void;
            if let Err(e) = evt.set_callback(CL_COMPLETE, hpc_core::memtrace_callback, ptr) {
                eprintln!("kernel callback failed: {e}");
                unsafe { Box::from_raw(ptr.cast::<hpc_core::CopyToken>()) }.finish();
            }
        }

        // Auf Kernel-Abschluss warten, InFlight -> Ready
        let ready_dst = dst_if.complete(evt);
        // Ping-Pong-Swap: neuer ping = ready_dst
        ping = ready_dst;
    }

    /* ---------- Ergebnis zurücklesen ---------- */
    let mut host_out = vec![0.0_f32; NX * NY];
    #[cfg(feature = "memtrace")]
    let rd_tok = trace_start(Dir::D2H, host_out.len() * std::mem::size_of::<f32>());

    queue.enqueue_read_buffer(
        ping.raw(), CL_BLOCKING, 0,
        cast_slice_mut(&mut host_out), &[],
    )?;
    queue.finish()?;

    #[cfg(feature = "memtrace")]
    rd_tok.finish();

    println!("Jacobi done, center value = {}", host_out[(NY/2) * NX + NX/2]);

    #[cfg(feature = "metrics")] summary();
    #[cfg(feature = "memtrace")] flush_csv();

    Ok(())
}