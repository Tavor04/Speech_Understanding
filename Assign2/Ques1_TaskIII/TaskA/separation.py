from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import os

# Load the model
model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

# Paths
input_mixture_path = "./mixed_vox2/test/mix_clean"
output_dir = "./estimated_sources"
os.makedirs(output_dir, exist_ok=True)

# Process all mixture files
for filename in os.listdir(input_mixture_path):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_mixture_path, filename)

        # Separate the file
        est_sources = model.separate_file(path=input_path)

        # Save the estimated sources
        torchaudio.save(os.path.join(output_dir, f"{filename[:-4]}_source1.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(output_dir, f"{filename[:-4]}_source2.wav"), est_sources[:, :, 1].detach().cpu(), 8000)

        print(f"[INFO] Processed and saved separated sources for {filename}")

