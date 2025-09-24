
# seed_db.py
import numpy as np
from database import save_face_record

# Fake embeddings (128-dim vector)
embedding = np.random.rand(128).astype(np.float32)

save_face_record("Alice", 0.95, [100, 100, 200, 200], embedding)
save_face_record("Bob", 0.87, [50, 50, 150, 150], embedding)
save_face_record("Unknown", 0.4, [300, 300, 400, 400], embedding)

print("✅ Sample data inserted.")