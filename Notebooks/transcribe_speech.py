import nemo.collections.asr as nemo_asr
import torch

def transcribe(audio_file):
    # Load pretrained Citrinet Chinese ASR model and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_zh_citrinet_1024_gamma_0_25")
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_quartznet15x5")
    asr_model = asr_model.to(device).eval()

    # Transcribe the audio file (expects a list of file paths)
    transcription = asr_model.transcribe([audio_file])
    print("Transcription:", transcription[0])

    # Save transcription to a text file
    with open("transcription_output.txt", "w", encoding="utf-8") as f:
        f.write(transcription[0])

if __name__ == "__main__":
    # Replace this with your audio file path
    audio_path = "p3ht_polymerization.wav"
    transcribe(audio_path)
