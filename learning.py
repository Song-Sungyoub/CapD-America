import os
import json
import torch
import torchaudio
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_metric
from tqdm import tqdm
import librosa

# 1. 데이터 로드 및 전처리
class SpeechDataset(Dataset):
    def __init__(self, audio_dir, label_dir, processor):
        self.audio_paths = []
        self.transcripts = []
        self.processor = processor

        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])

        assert len(audio_files) == len(label_files), "The number of audio files and label files must be the same."

        for audio_file, label_file in tqdm(iterable=zip(audio_files, label_files), desc="dataset append "):
            audio_path = os.path.join(audio_dir, audio_file)
            with open(os.path.join(label_dir, label_file), 'r', encoding="utf-8") as f:
                transcript = json.load(f)['fileName']
                
            self.audio_paths.append(audio_path)
            self.transcripts.append(transcript)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate).input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(transcript).input_ids
        return {"input_values": input_values, "labels": labels}

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
train_audio_dir = 'C:\\ai\\Training\\TS'
train_label_dir = 'C:\\ai\\Training\\TL'
val_audio_dir = 'C:\\ai\\Validation\\VS'
val_label_dir = 'C:\\ai\\Validation\\VL'
train_dataset = SpeechDataset(train_audio_dir, train_label_dir, processor)
val_dataset = SpeechDataset(val_audio_dir, val_label_dir, processor)

def collate_fn(batch):
    input_values = [item for item in batch]
    labels = [item for item in batch]
    input_values = processor.pad(input_values, return_tensors="pt", padding=True).input_values
    labels = processor.pad(labels, return_tensors="pt", padding=True).labels
    return {"input_values": input_values, "labels": labels}

def resample_audio(input_audio, original_sr, target_sr=16000):
    if original_sr != target_sr:
        return librosa.resample(input_audio, orig_sr=original_sr, target_sr=target_sr)
    return input_audio

train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 2. 모델 설정
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-lv60")
model.freeze_feature_extractor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 학습 및 평가 루프
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    total_loss = 0
    pbar = tqdm(iterable=train_loader, desc="epoch " + str(epoch + 1))
    
    for batch in pbar:
        try:
            #input_values = batch["input_values"].to(device)
            audio, sr = batch["input_values"]
            audio = resample_audio(audio.numpy(), sr, 16000)
            input_values = torch.tensor(audio).to(device)
            print(type(input_values))
            
            labels = batch["labels"].to(device)
            print(type(labels))
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            print(f"Error occurred for batch index {batch_idx}, skipping this batch. Error: {e}")
            continue
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
    
    # 4. validation 평가
    model.eval()
    val_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        pbar = tqdm(iterable=val_loader, desc="validation " + str(epoch + 1))
        
        for batch in pbar:
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = processor.batch_decode(predicted_ids)
                references = processor.batch_decode(labels, group_tokens=False)
                all_predictions.extend(predictions)
                all_references.extend(references)
            except Exception as e:
                print(f"Error occurred for batch index {batch_idx}, skipping this batch. Error: {e}")
                continue
    
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")
    
    wer = load_metric("wer")
    wer_score = wer.compute(predictions=all_predictions, references=all_references)
    print(f"Word Error Rate: {wer_score}")

# 5. 모델 저장
model.save_pretrained("fine-tuned-wav2vec1")
processor.save_pretrained("fine-tuned-wav2vec1")
