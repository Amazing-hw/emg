"""Verify all 4 methods import and forward-pass correctly."""
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print()

# Method 1: Conformer
print("=" * 60)
print("Method 1: ConformerGestureArchitecture")
sys.path.insert(0, "D:/emg/method1_conformer")
from emg_transfer.networks import ConformerGestureArchitecture, Emg2PoseTdsGestureArchitecture, DiscreteGesturesArchitecture

m1 = ConformerGestureArchitecture(num_layers=4)
x1 = torch.randn(2, 16, 4000)
y1 = m1(x1)
print(f"  Output: {y1.shape}")
print(f"  Params: {sum(p.numel() for p in m1.parameters()):,}")
print(f"  left_context: {m1.left_context}, stride: {m1.stride}")

# Method 1b: TDS baseline (also in method1 for comparison)
m1b = Emg2PoseTdsGestureArchitecture()
y1b = m1b(x1)
print(f"  TDS baseline output: {y1b.shape}")
print(f"  TDS params: {sum(p.numel() for p in m1b.parameters()):,}")

# Method 2: CTC
print("\n" + "=" * 60)
print("Method 2: CtcGestureArchitecture + CifGestureArchitecture")
sys.path.insert(0, "D:/emg/method2_ctc_cif")
from emg_transfer.networks import CtcGestureArchitecture, CifGestureArchitecture

m2_ctc = CtcGestureArchitecture(num_layers=4)
x2 = torch.randn(2, 16, 4000)
y2 = m2_ctc(x2)
print(f"  CTC output shape: {y2.shape}")
input_lens = m2_ctc.get_input_lengths(x2)
print(f"  Blank ID: {m2_ctc.blank_id}, Input lengths: {input_lens}")

m2_cif = CifGestureArchitecture(num_layers=4)
y2c = m2_cif(x2)
print(f"  CIF output: logits={y2c['logits'].shape}, alpha={y2c['alpha'].shape}, num_events={y2c['num_events']}")

# Method 3: SSL
print("\n" + "=" * 60)
print("Method 3: Wav2Vec2 + HuBERT pretraining models")
sys.path.insert(0, "D:/emg/method3_ssl_pretrain")
from emg_transfer.ssl_networks import Wav2Vec2Model, HubertModel, EmgFeatureEncoder

m3_w2v = Wav2Vec2Model(num_conformer_layers=4)
x3 = torch.randn(2, 16, 4000)
c, q, mask, ppl = m3_w2v(x3)
print(f"  Wav2Vec2: c={c.shape}, q={q.shape}, mask_ratio={mask.float().mean():.3f}, ppl={ppl:.2f}")

m3_hub = HubertModel(num_conformer_layers=4)
logits, mask_h, _ = m3_hub(x3)
print(f"  HuBERT: logits={logits.shape}, mask_ratio={mask_h.float().mean():.3f}")

# SSL Fine-tuning
from emg_transfer.ssl_finetune import SslGestureArchitecture
m3_ft = SslGestureArchitecture(num_conformer_layers=4)
y3_ft = m3_ft(x3)
print(f"  SSL FT output: {y3_ft.shape}")

# Method 4: Multi-task
print("\n" + "=" * 60)
print("Method 4: Multi-task Gesture + Joint Angle")
sys.path.insert(0, "D:/emg/method4_multitask")
from emg_transfer.multitask_networks import MultiTaskGesturePoseArchitecture, JointAngleToGestureMapper

m4 = MultiTaskGesturePoseArchitecture(num_layers=4)
y4g = m4(x1, task="gesture")
y4j = m4(x1, task="joint")
y4b = m4(x1, task="both")
print(f"  Gesture head: {y4g.shape}")
print(f"  Joint head: {y4j.shape}")
print(f"  Both: gesture={y4b[0].shape}, joint={y4b[1].shape}")
print(f"  Params: {sum(p.numel() for p in m4.parameters()):,}")

# Test weak label generation
ja = torch.randn(2, 20, 50)
weak, conf = JointAngleToGestureMapper.infer_gesture_from_angles(ja)
print(f"  Weak labels: {weak.shape}, active ratio={weak.sum()/weak.numel():.3f}")

print("\n" + "=" * 60)
print("ALL METHODS VERIFIED SUCCESSFULLY")
