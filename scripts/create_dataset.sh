#
#data_path="/home/vieira/Desktop/datasets/mini_FFHQ/"
#psf_path="psf/simulation/256/random.npy"
#output_path="/home/vieira/Desktop/datasets/SelfSupervisedLensless/mini_FFHQ_256/"
#noise_std=0.005
#
#
#python scripts/create_dataset.py --data_path $data_path --psf_path $psf_path --output_path $output_path --noise_std $noise_std


#data_path="/home/vieira/datasets/mini_FFHQ"
#psf_path="psf/simulation/256/radial_13per.npy"
#output_path="/home/vieira/datasets/mini_FFHQ_256/radial/"
#noise_std=0.005
#
#python scripts/create_dataset.py --data_path $data_path --psf_path $psf_path --output_path $output_path --noise_std $noise_std


data_path="/home/vieira/datasets/mini_FFHQ/"
psf_path="psf/simulation/256/radial_6per.npy"
output_path="/home/vieira/datasets/SelfSupervisedLensless/simulation/"
noise_std=0.005


python scripts/create_dataset.py --data_path $data_path --psf_path $psf_path --output_path $output_path --noise_std $noise_std