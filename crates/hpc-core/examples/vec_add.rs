// 2025 Thomas Bicanic – MIT License

use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::ClError;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
};

#[cfg(feature = "metrics")]
use hpc_core::summary;
#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};

fn main() -> Result<(), ClError> {
    /* ---------- 1. OpenCL-Setup ---------------------------------- */
    let platform   = get_platforms()?.remove(0);
    let device_id  = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device     = Device::new(device_id);
    let context    = Context::from_device(&device)?;
    let queue      = CommandQueue::create(
        &context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    /* ---------- 2. Hostdaten ------------------------------------- */
    let n           = 1 << 22;                       // 1 048 576 Elemente
    let size_bytes  = n * std::mem::size_of::<f32>(); // 4 MiB
    let h_a         = vec![1.0_f32; n];
    let h_b         = vec![2.0_f32; n];
    let mut h_out   = vec![0.0_f32; n];

    /* ---------- 3. Device-Buffer anlegen ------------------------- */
    let mut a_dev: Buffer<f32>  =
        Buffer::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;
    let mut b_dev: Buffer<f32>  =
        Buffer::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;
    let mut out_dev: Buffer<f32> =
        Buffer::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;

    /* ---------- 4. Host → Device  (ein Sammel-Token) ------------- */
    /* ---------- 4. Host→Device – zwei seriell getrennte Tokens ------- */
use opencl3::types::{CL_BLOCKING};   // oben in den use-Blöcken ergänzen

// --- Kopie A ------------------------------------------------------
#[cfg(feature="memtrace")]
let tok_h2d_a = trace_start(Dir::H2D, size_bytes);     // Token A starten

queue.enqueue_write_buffer(
    &mut a_dev,
    CL_BLOCKING,         // <-- blockierend!
    0,
    cast_slice(&h_a),
    &[],
)?;                       // Funktion kehrt erst zurück, wenn DMA fertig ist

#[cfg(feature="memtrace")]
tok_h2d_a.finish();        // Token A schließen – 1. H2D-Zeile

// --- Kopie B ------------------------------------------------------
#[cfg(feature="memtrace")]
let tok_h2d_b = trace_start(Dir::H2D, size_bytes);     // Token B starten

queue.enqueue_write_buffer(
    &mut b_dev,
    CL_BLOCKING,          // wieder blockierend
    0,
    cast_slice(&h_b),
    &[],
)?;

#[cfg(feature="memtrace")]
tok_h2d_b.finish();        // Token B schließen – 2. H2D-Zeile
/* ---------------------------------------------------------------- */                                  // genau **eine** H2D-Zeile

    /* ---------- 5. Kernel – ein Token --------------------------- */
    #[cfg(feature = "memtrace")]
    let tok_kernel = trace_start(Dir::Kernel, 0);

    let src     = include_str!("../examples/vec_add.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "vec_add")?;
    kernel.set_arg(0, &a_dev)?;   // Buffer implementiert KernelArg
    kernel.set_arg(1, &b_dev)?;
    kernel.set_arg(2, &out_dev)?;

    let global = [n, 1, 1];
    queue.enqueue_nd_range_kernel(
        kernel.get(), 1,
        std::ptr::null(), global.as_ptr(),
        std::ptr::null(), &[])?;
    queue.finish()?;                                         // Kernel fertig

    #[cfg(feature = "memtrace")]
    tok_kernel.finish();                                     // eine Kernel-Zeile

    /* ---------- 6. Device → Host – ein Token -------------------- */
    #[cfg(feature = "memtrace")]
    let tok_d2h = trace_start(Dir::D2H, size_bytes);

    queue.enqueue_read_buffer(
        &mut out_dev, CL_BLOCKING, 0,
        cast_slice_mut(&mut h_out), &[])?;                   // blockierend

    #[cfg(feature = "memtrace")]
    tok_d2h.finish();                                       // eine D2H-Zeile

    /* ---------- 7. Verifizieren & Ausgabe ----------------------- */
    assert!(h_out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    println!("vec_add OK, first element = {}", h_out[0]);

    #[cfg(feature = "metrics")]
    summary();
    #[cfg(feature = "memtrace")]
    flush_csv();                                             // schreibt memtrace.csv

    Ok(())
}
