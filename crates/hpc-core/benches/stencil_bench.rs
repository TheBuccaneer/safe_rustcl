use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use hpc_core::{ClError, GpuBuffer, Queued, Ready};
use bytemuck::cast_slice;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
};

use std::time::Duration;

const NX: usize = 1024;
const NY: usize = 1024;
const N_BYTES: usize = NX * NY * std::mem::size_of::<f32>();
const N_ITERS: usize = 10;

fn bench_stencil(c: &mut Criterion) {
    c.bench_function("jacobi_1024x1024_10iter", |b| {
        b.iter_batched(
            /* ---------- Setup ---------- */
            || {
                let platform = get_platforms().unwrap().remove(0);
                let dev_id   = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
                let device   = Device::new(dev_id);
                let context  = Context::from_device(&device).unwrap();
                let queue    = CommandQueue::create(
                    &context, device.id(), CL_QUEUE_PROFILING_ENABLE,
                ).unwrap();

                // Kernel
                let src = include_str!("../examples/stencil.cl");
                let program = Program::create_and_build_from_source(&context, src, "").unwrap();
                let mut kernel = Kernel::create(&program, "jacobi").unwrap();
                kernel.set_arg(2, &(NX as i32)).unwrap();

                // Ping initialisieren (Ready)
                let init = vec![1.0_f32; NX * NY];
                let (if_buf, g) = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap()
                    .enqueue_write(&queue, cast_slice(&init)).unwrap();
                let ping_ready: GpuBuffer<Ready> = if_buf.into_ready(g);

                (context, queue, kernel, ping_ready)
            },
            /* ---------- Mess‑Body ---------- */
            |(context, queue, mut kernel, mut ping)| {
                for _ in 0..N_ITERS {
                    // Ziel‑Puffer: Queued → InFlight
                    let mut dst_if = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap()
                        .launch();

                    kernel.set_arg(0, ping.raw()).unwrap();
                    kernel.set_arg(1, dst_if.raw_mut()).unwrap();

                    let global = [NX, NY, 1];
                    let evt = queue.enqueue_nd_range_kernel(
                        kernel.get(), 2,
                        std::ptr::null(), global.as_ptr(),
                        std::ptr::null(), &[],
                    ).unwrap();
                    evt.wait().unwrap();

                    // InFlight -> Ready
                    let ready_dst = dst_if.complete(evt);
                    // Ping‑Pong‑Swap
                    ping = ready_dst;
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn criterion_config() -> Criterion {
    Criterion::default()
        // erlaube bis zu 30 s Messdauer
        .measurement_time(Duration::from_secs(20))
        // behalte sample_size=100 (Default)
}


// Diese Zeilen sind notwendig, damit Criterion den Benchmark ausführt
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_stencil
}
criterion_main!(benches);
