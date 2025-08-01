// Benchmark for Jacobi 4-point stencil – **tuned for low-jitter runs**
// ────────────────────────────────────────────────────────────────────────────────
// Warm-up, sample size, measurement time and throughput follow the guidelines
// discussed in the chat: 3 s warm-up, 30 samples, 10 s measurement window.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use hpc_core::{GpuBuffer, Queued, Ready};
use bytemuck::cast_slice;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
    types::CL_BLOCKING,
};

use std::ptr;
use std::time::Duration;

const NX: usize = 1024;
const NY: usize = 1024;
const N_BYTES: usize = NX * NY * std::mem::size_of::<f32>();
const N_ITERS: usize = 10; // kernel sweeps per benchmark sample

// ───────────────────────────────────────────────────────────── benchmark fn ────
fn bench_stencil(c: &mut Criterion) {
    // 1. Create a benchmark group so we can attach throughput metadata
    let mut g = c.benchmark_group("jacobi4");

    // ≈ 3 buffers (src, dst, ping-pong) transferred per sweep
    g.throughput(Throughput::Bytes((NX * NY * 4 * 3) as u64));

    g.bench_function("raw_jacobi_1024x1024_10iter", |b| {
        b.iter_batched(
            || {
                // Setup wie gehabt
                let platform = get_platforms().unwrap().remove(0);
                let dev_id   = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
                let device   = Device::new(dev_id);
                let ctx      = Context::from_device(&device).unwrap();
                let queue    = CommandQueue::create(&ctx, device.id(), CL_QUEUE_PROFILING_ENABLE).unwrap();
                let src      = include_str!("../examples/stencil.cl");
                let program  = Program::create_and_build_from_source(&ctx, src, "").unwrap();
                let kern = Kernel::create(&program, "jacobi").unwrap();

                // Buffers manuell anlegen + initialisieren
                let mut buf_a = Buffer::<f32>::create(&ctx, CL_MEM_READ_WRITE, NX*NY, ptr::null_mut()).unwrap();
                let buf_b = Buffer::<f32>::create(&ctx, CL_MEM_READ_WRITE, NX*NY, ptr::null_mut()).unwrap();
                // init in buf_a
                queue.enqueue_write_buffer(&mut buf_a, CL_BLOCKING, 0, cast_slice(&vec![1.0_f32; NX*NY]), &[]).unwrap();
                (queue, kern, buf_a, buf_b)
            },
            |(queue, kern, mut src_buf, mut dst_buf)| {
                // 10 Iterationen Raw
                for _ in 0..N_ITERS {
                    // Set all 4 kernel arguments for each iteration
                    kern.set_arg(0, &src_buf).unwrap();
                    kern.set_arg(1, &dst_buf).unwrap();
                    kern.set_arg(2, &(NX as i32)).unwrap(); // width
                    kern.set_arg(3, &(NY as i32)).unwrap(); // height ← This was missing!
                    
                    let evt = queue
                        .enqueue_nd_range_kernel(kern.get(), 2, std::ptr::null(), [NX,NY,1].as_ptr(), std::ptr::null(), &[])
                        .unwrap();
                    evt.wait().unwrap();
                    std::mem::swap(&mut src_buf, &mut dst_buf);
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("jacobi_1024x1024_10iter", |b| {
        b.iter_batched(
            /* ----------- Setup (executed once per sample) ----------- */
            || {
                // OpenCL boilerplate
                let platform  = get_platforms().unwrap().remove(0);
                let dev_id    = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
                let device    = Device::new(dev_id);
                let context   = Context::from_device(&device).unwrap();
                let queue     = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE).unwrap();

                // Build kernel
                let src      = include_str!("../examples/stencil.cl");
                let program  = Program::create_and_build_from_source(&context, src, "").unwrap();
                let kern = Kernel::create(&program, "jacobi").unwrap();

                // Initialise ping buffer (Ready)
                let init          = vec![1.0_f32; NX * NY];
                let (if_buf, g)   = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap()
                    .enqueue_write(&queue, cast_slice(&init)).unwrap();
                let ping_ready: GpuBuffer<Ready> = if_buf.into_ready(g);

                (context, queue, kern, ping_ready)
            },
            /* ------------------- Measured body --------------------- */
            |(context, queue, kern, mut ping)| {
                for _ in 0..N_ITERS {
                    // dst: Queued → InFlight
                    let mut dst_if = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap().launch();

                    kern.set_arg(0, ping.raw()).unwrap();
                    kern.set_arg(1, dst_if.raw_mut()).unwrap();
                    kern.set_arg(2, &(NX as i32)).unwrap(); // width
                    kern.set_arg(3, &(NY as i32)).unwrap(); // height ← Make sure this is set every iteration

                    // global ND-range
                    let global = [NX, NY, 1];
                    let evt = queue
                        .enqueue_nd_range_kernel(kern.get(), 2, std::ptr::null(), global.as_ptr(), std::ptr::null(), &[])
                        .unwrap();
                    evt.wait().unwrap();

                    // InFlight → Ready, swap ping-pong
                    let ready_dst = dst_if.complete(evt);
                    ping = ready_dst;
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────── Criterion config ──
fn criterion_config() -> Criterion {
    Criterion::default()
        // 3 s warm-up to heat up GPU / caches
        .warm_up_time(Duration::from_secs(3))
        // 10 s measurement window per sample
        .measurement_time(Duration::from_secs(10))
        // 30 statistically independent samples
        .sample_size(30)
        // allow `cargo bench --bench ... -- <extra args>` to override settings
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_stencil
}
criterion_main!(benches);