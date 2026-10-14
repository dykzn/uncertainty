"""Match NVIDIA GPU UUIDs to EGL devices rather than assuming equal ordinals."""
import argparse
import csv
import ctypes
import io
import json
import os
import subprocess


def detect():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run([os.sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True)
    cards = [{"index": int(r[0]), "uuid": r[1].strip(), "name": r[2].strip(),
              "total_MiB": int(r[3]), "used_MiB": int(r[4])} for r in csv.reader(io.StringIO(output))]
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    devices_fn = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    attr_fn = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    uuid_fn = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    uuid_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    devices = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    if not devices_fn(64, devices, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    mappings = {}
    for i in range(count.value):
        ordinal = ctypes.c_ssize_t(-1)
        if not attr_fn(devices[i], 0x323A, ctypes.byref(ordinal)):
            continue
        raw = (ctypes.c_ubyte * 16)()
        if uuid_fn(raw, ordinal.value) != 0:
            continue
        h = bytes(raw).hex()
        uuid = "GPU-" + "-".join([h[:8], h[8:12], h[12:16], h[16:20], h[20:]])
        mappings[uuid] = i
    for card in cards:
        card["egl"] = mappings.get(card["uuid"])
    return cards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--egl-for-visible", action="store_true")
    args = parser.parse_args()
    cards = detect()
    if args.egl_for_visible:
        wanted = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        card = next(c for c in cards if c["uuid"] == wanted or str(c["index"]) == wanted)
        if card["egl"] is None:
            raise RuntimeError("Selected GPU has no EGL renderer")
        print(card["egl"])
    else:
        print(json.dumps(cards, indent=2))
