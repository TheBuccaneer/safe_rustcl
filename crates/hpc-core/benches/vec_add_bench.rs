use criterion::{Criterion, criterion_group, criterion_main};
use hpc_core::{GpuBuffer, Queued};
use bytemuck::cast_slice;
use opencl3::{
    context::Context, command_queue::CommandQueue, program::Program, kernel::Kernel,
    platform::get_platforms, device::{Device, CL_DEVICE_TYPE_GPU}, types::CL_BLOCKING,
};

fn bench_vec_add(c: &mut Criterion) {
    c.bench_function("vec_add_1KiB", |b| {
        b.iter(|| {
            let n = 256; // 256 * 4B = 1 KiB
            let h_a = vec![1.0_f32; n];
            let h_b = vec![2.0_f32; n];
            let mut h_out = vec![0.0_f32; n];

            let platform = get_platforms().unwrap().remove(0);
            let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
            let device = Device::new(device_ids[0]);
            let context = Context::from_device(&device).unwrap();
            let queue = CommandQueue::create(&context, device.id(), 0).unwrap();

            let mut a = GpuBuffer::<Queued>::new(&context, n * 4).unwrap();
            let mut b = GpuBuffer::<Queued>::new(&context, n * 4).unwrap();
            let out = GpuBuffer::<Queued>::new(&context, n * 4).unwrap();

            queue.enqueue_write_buffer(a.raw_mut(), CL_BLOCKING, 0, cast_slice(&h_a), &[]).unwrap();
            queue.enqueue_write_buffer(b.raw_mut(), CL_BLOCKING, 0, cast_slice(&h_b), &[]).unwrap();
            queue.finish().unwrap();

            let src = include_str!("../examples/vec_add.cl");
            let program = Program::create_and_build_from_source(&context, src, "").unwrap();
            let kernel = Kernel::create(&program, "vec_add").unwrap();

            kernel.set_arg(0, a.raw()).unwrap();
            kernel.set_arg(1, b.raw()).unwrap();
            kernel.set_arg(2, out.raw()).unwrap();

            let global = [n, 1, 1];
            let _evt = queue.enqueue_nd_range_kernel(kernel.get(), 1, std::ptr::null(), global.as_ptr(), std::ptr::null(), &[]).unwrap();
            queue.finish().unwrap();

            queue.enqueue_read_buffer(out.raw(), CL_BLOCKING, 0, bytemuck::cast_slice_mut(&mut h_out), &[]).unwrap();

            queue.finish().unwrap();

            assert!((h_out[0] - 3.0).abs() < 1e-6);
        });
    });
}

// Diese Zeilen sind notwendig, damit Criterion den Benchmark ausführt
criterion_group!(benches, bench_vec_add);
criterion_main!(benches);
