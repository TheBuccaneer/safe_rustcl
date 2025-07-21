// 2025 Thomas Bicanic
//
// This software is released under the MIT License.
// See LICENSE file in repository root for full license text.

use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued};

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
    types::CL_BLOCKING,
};

#[cfg(feature = "metrics")]
use hpc_core::summary;                      // Mean & P95

#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};  // ← Alias gegen Namenskollision

fn main() -> Result<(), ClError> {


    // ---------- 1. Context & Queue ----------------------------------
    let platform   = get_platforms()?.remove(0);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
    let device     = Device::new(device_ids[0]);
    let context    = Context::from_device(&device)?;
    let queue = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    // ---------- 2. Host‑Daten ---------------------------------------
    let n = 1 << 20;
    let h_a = vec![1.0_f32; n];
    let h_b = vec![2.0_f32; n];
    let mut h_out = vec![0.0_f32; n];

    // ---------- 3. Device‑Buffer ------------------------------------
    let a_dev = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;
    let b_dev = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;
    let out_dev = GpuBuffer::<Queued>::new(&context, n * std::mem::size_of::<f32>())?;

    // Host → Device
    let (a_inflight, g_a) = a_dev.enqueue_write(&queue, cast_slice(&h_a))?;
    let (b_inflight, g_b) = b_dev.enqueue_write(&queue, cast_slice(&h_b))?;
    let a_ready = a_inflight.into_ready(g_a);
    let b_ready = b_inflight.into_ready(g_b);

    // ---------- 4. Kernel kompilieren & setzen ----------------------
     #[cfg(feature = "memtrace")]
    let kernel_tok = Box::new(trace_start(Dir::Kernel, 0));

    
    let src = include_str!("vec_add.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel = Kernel::create(&program, "vec_add")?;
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

    #[cfg(feature = "memtrace")]
{
    use opencl3::event::CL_COMPLETE;          // Status-Konstante für „fertig“

    // 2) Pointer auf token_box erzeugen
    let user_ptr = Box::into_raw(kernel_tok) as *mut std::ffi::c_void;

    // 3) Callback setzen – memtrace_callback liegt schon in lib.rs
    if let Err(e) = evt_kernel.set_callback(CL_COMPLETE, hpc_core::memtrace_callback, user_ptr) {
        eprintln!("kernel callback failed: {e}");
        // Fallback: Box zurückholen und finish() sofort ausführen
        unsafe { Box::from_raw(user_ptr as *mut hpc_core::CopyToken) }.finish();
    }
}

    // ---------- 5. Kernel‑Profiling ---------------------------------
    evt_kernel.wait()?;
    let k_start = evt_kernel.profiling_command_start()?;
    let k_end   = evt_kernel.profiling_command_end()?;
    let kernel_us = (k_end.saturating_sub(k_start)) / 1_000;
    println!("Kernel execution: {} µs", kernel_us);

    // ---------- 6. Device → Host kopieren ---------------------------
    #[cfg(feature = "memtrace")]
    let tok_read = trace_start(Dir::D2H, h_out.len() * std::mem::size_of::<f32>());

    queue.enqueue_read_buffer(
        out_dev.raw(),
        CL_BLOCKING,
        0,
        cast_slice_mut(&mut h_out),
        &[],
    )?;
    queue.finish()?;

    #[cfg(feature = "memtrace")]
    tok_read.finish();

    // ---------- 7. Verifizieren -------------------------------------
    assert!(h_out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    println!("vec_add OK, first element = {}", h_out[0]);

    // ---------- 8. Ausgaben -----------------------------------------
    #[cfg(feature = "metrics")]
    summary();

    #[cfg(feature = "memtrace")]
    flush_csv();

    Ok(())
}
