import sys
import traceback
sys.path.insert(0, 'simulator')
from generate_pairs import build_room_simulation

try:
    build_room_simulation()
    print("Success")
except Exception:
    traceback.print_exc()
