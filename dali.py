from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

image_dir = "dali_images"
max_batch_size = 8

def show_images(image_batch):
    columns = 4
    rows = (max_batch_size + 1) // (columns)
    fig = plt.figure(figsize=(24, (24 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))
    plt.tight_layout()
    plt.savefig('img_group.png')

@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device="cpu")

    return images, labels

@pipeline_def
def shuffled_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=21)
    images = fn.decoders.image(jpegs, device="cpu")

    return images, labels

@pipeline_def
def rotated_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=21)
    images = fn.decoders.image(jpegs, device="cpu")
    angle = fn.random.uniform(range=(-10.0, 10.0))
    rotated_images = fn.rotate(images, angle=angle, fill_value=0)

    return rotated_images, labels

@pipeline_def
def random_rotated_gpu_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=21)
    images = fn.decoders.image(jpegs, device="mixed")
    angle = fn.random.uniform(range=(-10.0, 10.0))
    rotated_images = fn.rotate(images.gpu(), angle=angle, fill_value=0)

    return rotated_images, labels

# pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
# pipe.build()

pipe = random_rotated_gpu_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0, seed=1234)
pipe.build()

pipe_out = pipe.run()
# print(pipe_out)

images, labels = pipe_out
show_images(images.as_cpu())
