"""
Background Removal — Model 1
BiRefNet + IS-Net Ensemble  |  v5.2
"""

import streamlit as st
import gc, io, zipfile, time, warnings
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BG Remover — Model 1",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0C0E14; color: #D4D8E2; }

[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1C1F29;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: #7A8299 !important; font-size: 13px !important; }

h1,h2,h3 { color: #E8ECF5 !important; }

[data-testid="metric-container"] {
    background: #13161E;
    border: 1px solid #1C1F29;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label { color: #404660 !important; font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00C9A7 !important; font-size: 24px !important; font-weight: 700 !important; }

.stButton > button {
    background: #00C9A7;
    color: #001A16;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 24px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #00E8C0; transform: translateY(-1px); }

[data-testid="stFileUploader"] {
    background: #13161E;
    border: 2px dashed #1C2840;
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover { border-color: #00C9A7; }

.stSlider > div > div > div > div { background-color: #00C9A7 !important; }
.stSelectbox > div > div {
    background: #13161E !important;
    border: 1px solid #1C1F29 !important;
    color: #D4D8E2 !important;
    border-radius: 8px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00C9A7, #00E8C0);
    border-radius: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #13161E;
    border-radius: 10px;
    padding: 3px;
    gap: 3px;
    border: 1px solid #1C1F29;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #404660;
    border-radius: 7px;
    font-weight: 500;
    font-size: 13px;
    padding: 7px 20px;
}
.stTabs [aria-selected="true"] { background: #00C9A7 !important; color: #001A16 !important; }

hr { border: none; border-top: 1px solid #1C1F29; margin: 16px 0; }

.panel-label {
    font-size: 10px;
    font-weight: 700;
    color: #404660;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    text-align: center;
    margin-bottom: 6px;
}

.chip-ok {
    display: inline-block;
    background: rgba(0,201,167,0.1);
    border: 1px solid rgba(0,201,167,0.3);
    color: #00C9A7;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 11px;
    font-weight: 600;
}
.chip-err {
    display: inline-block;
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.3);
    color: #F87171;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 11px;
    font-weight: 600;
}

.info-tag {
    background: #13161E;
    border: 1px solid #1C1F29;
    border-left: 3px solid #00C9A7;
    border-radius: 7px;
    padding: 9px 12px;
    margin: 5px 0;
    font-size: 12px;
    color: #7A8299;
}
.info-tag b { color: #00C9A7; }

.checker {
    background-image:
        linear-gradient(45deg, #181B24 25%, transparent 25%),
        linear-gradient(-45deg, #181B24 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #181B24 75%),
        linear-gradient(-45deg, transparent 75%, #181B24 75%);
    background-size: 14px 14px;
    background-position: 0 0, 0 7px, 7px -7px, -7px 0;
    background-color: #13161E;
    border-radius: 8px;
}

.dl-teal > button {
    background: #00C9A7 !important;
    color: #001A16 !important;
    font-size: 12px !important;
    padding: 7px 16px !important;
}
.dl-teal > button:hover { background: #00E8C0 !important; }

.stSpinner > div { border-top-color: #00C9A7 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1C1F29; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

for k, v in {"results": [], "models_ok": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


class Config:
    BIREFNET_MODEL_ID   = "ZhengPeng7/BiRefNet"
    BIREFNET_INPUT_SIZE = (1024, 1024)
    BIREFNET_MEAN       = [0.485, 0.456, 0.406]
    BIREFNET_STD        = [0.229, 0.224, 0.225]
    MORPH_OPEN_KERNEL   = 3
    MORPH_CLOSE_KERNEL  = 7
    GF_RADIUS           = 10
    GF_EPS              = 1e-4
    BILATERAL_D         = 9
    BILATERAL_SIGMA_C   = 80
    BILATERAL_SIGMA_S   = 80
    MIN_COMPONENT_RATIO = 0.001

CFG = Config()


@st.cache_resource(show_spinner=False)
def load_birefnet(device: str, input_size: int):
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        model = (
            AutoModelForImageSegmentation
            .from_pretrained(CFG.BIREFNET_MODEL_ID, trust_remote_code=True)
            .to(device)
            .eval()
        )
        # CUDA dtype fix: cast to float32 to avoid Half/Float mismatch
        if device == "cuda":
            model = model.float()
        tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CFG.BIREFNET_MEAN, std=CFG.BIREFNET_STD),
        ])
        return model, tf
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_rembg():
    try:
        from rembg import remove, new_session
        return remove, new_session("isnet-general-use")
    except Exception:
        return None, None


def predict_birefnet(img, model, tf, device):
    if model is None: return None
    import torch
    w, h = img.size
    tensor = tf(img.convert("RGB")).unsqueeze(0).to(device)
    # ensure float32 on CUDA
    if device == "cuda":
        tensor = tensor.float()
    with torch.no_grad():
        preds = model(tensor)[-1].sigmoid().cpu()
    mask = (preds[0].squeeze().numpy() * 255).astype(np.uint8)
    del tensor, preds; gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def predict_rembg(img, remove_fn, session):
    if session is None: return None
    import torch
    result = remove_fn(img, session=session)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    arr = np.array(result)
    return arr[:, :, 3] if result.mode == "RGBA" else np.array(result.convert("L"))


def merge_masks(masks, w_bi):
    w = {"birefnet": w_bi, "rembg": 1.0 - w_bi}
    av = {k: v for k, v in masks.items() if v is not None}
    if not av: raise ValueError("All engines failed.")
    if len(av) == 1: return list(av.values())[0]
    total = sum(w[k] for k in av)
    acc = None
    for name, m in av.items():
        ww = w[name] / total
        mf = m.astype(np.float32) / 255.0
        acc = mf * ww if acc is None else acc + mf * ww
    return (acc * 255).astype(np.uint8)


def _gf(guide, src, r, eps):
    k = (2*r+1, 2*r+1)
    N = cv2.boxFilter(np.ones_like(guide), -1, k)
    mI = cv2.boxFilter(guide, -1, k) / N
    mp = cv2.boxFilter(src, -1, k) / N
    mIp= cv2.boxFilter(guide*src, -1, k) / N
    mII= cv2.boxFilter(guide**2, -1, k) / N
    a  = (mIp - mI*mp) / (mII - mI**2 + eps)
    b  = mp - a*mI
    return np.clip(cv2.boxFilter(a,-1,k)/N * guide + cv2.boxFilter(b,-1,k)/N, 0, 1)


def refine_mask(rgb, raw):
    m = raw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG.MORPH_OPEN_KERNEL,)*2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG.MORPH_CLOSE_KERNEL,)*2))
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    ref   = _gf(guide, m.astype(np.float32)/255., CFG.GF_RADIUS, CFG.GF_EPS)
    binary= (ref > .5).astype(np.float32)
    zone  = (ref > .3) & (ref < .7)
    r     = binary.copy(); r[zone] = ref[zone]
    r8    = cv2.bilateralFilter((r*255).astype(np.uint8), CFG.BILATERAL_D, CFG.BILATERAL_SIGMA_C, CFG.BILATERAL_SIGMA_S)
    b2    = (r8 > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b2)
    minpx = r8.shape[0]*r8.shape[1]*CFG.MIN_COMPONENT_RATIO
    clean = np.zeros_like(b2)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= minpx: clean[labels==i] = 1
    return (clean * r8).astype(np.uint8)


def apply_mask(img, mask):
    rgba = img.convert("RGBA")
    r, g, b, _ = rgba.split()
    return Image.merge("RGBA", (r, g, b, Image.fromarray(mask, "L")))


def checker_composite(rgba, tile=14):
    w, h = rgba.size
    bg = Image.new("RGB", (w, h))
    px = bg.load()
    dark, light = (18, 21, 30), (26, 30, 42)
    for y in range(h):
        for x in range(w):
            px[x, y] = light if ((x//tile)+(y//tile))%2==0 else dark
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def pil2bytes(img, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, format=fmt); return buf.getvalue()


def run_pipeline(pil_image, bi_model, bi_tf, bi_device, rm_fn, rm_sess, w_bi):
    t0 = time.time()
    rgb = np.array(pil_image.convert("RGB"))
    masks = {
        "birefnet": predict_birefnet(pil_image, bi_model, bi_tf, bi_device),
        "rembg":    predict_rembg(pil_image, rm_fn, rm_sess),
    }
    merged  = merge_masks(masks, w_bi)
    refined = refine_mask(rgb, merged)
    result  = apply_mask(pil_image, refined)
    return result, refined, time.time() - t0


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Settings")
    st.markdown("---")
    st.markdown("**BiRefNet device**")
    bi_device = st.selectbox("Device", ["cpu", "cuda"], index=0,
        help="CPU recommended — avoids dtype errors on most GPUs")
    bi_size = st.selectbox("Input resolution", [512, 768, 1024], index=2)

    st.markdown("---")
    st.markdown("**Ensemble weights**")
    w_bi = st.slider("BiRefNet weight", 0.0, 1.0, 0.60, 0.05)
    st.caption(f"rembg weight: **{round(1-w_bi,2)}**")
    st.markdown("---")

    if st.button("Load models"):
        load_birefnet.clear(); load_rembg.clear()
        with st.spinner("Loading…"):
            _m, _t = load_birefnet(bi_device, bi_size)
            _r, _s = load_rembg()
        if _m and _s: st.success("Both models ready.")
        elif _m:      st.warning("rembg failed.")
        elif _s:      st.warning("BiRefNet failed.")
        else:         st.error("Both models failed.")

    st.markdown("---")
    st.markdown(
        '<div class="info-tag"><b>BiRefNet</b> — Fine edges & detail</div>'
        '<div class="info-tag"><b>rembg IS-Net</b> — Fast wide-domain</div>'
        '<div class="info-tag" style="opacity:.4;border-left-color:#1C1F29">SAM2 — Disabled</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px;color:#1C1F29;text-align:center;margin-top:20px">'
        'v5.2 · Ahmed Nasser · Graduation Project</p>',
        unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin:0 0 4px;font-size:22px;font-weight:700;color:#E8ECF5">Background Removal</h2>'
    '<p style="color:#404660;font-size:13px;margin:0 0 20px">Model 1 · v5.2 · BiRefNet + IS-Net Ensemble</p>',
    unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_up, tab_batch = st.tabs(["Process", "Batch download"])


# ══════════════════════════════════════════════════
# TAB 1 — PROCESS
# ══════════════════════════════════════════════════
with tab_up:
    col_left, col_right = st.columns([1, 2], gap="medium")

    with col_left:
        st.markdown("##### Upload")
        uploaded = st.file_uploader(
            "Drop images",
            type=["jpg","jpeg","png","webp","bmp","tiff"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"{len(uploaded)} file(s) selected")

        btn = st.button("Remove backgrounds", disabled=not bool(uploaded))

        if btn and uploaded:
            bi_model, bi_tf = load_birefnet(bi_device, bi_size)
            rm_fn, rm_sess  = load_rembg()

            if bi_model is None and rm_sess is None:
                st.error("Both models failed. Run Cell 2 in the notebook to install dependencies.")
            else:
                st.session_state.results = []
                prog = st.progress(0)
                status = st.empty()
                total = len(uploaded)

                for i, f in enumerate(uploaded):
                    status.markdown(
                        f'<span style="color:#00C9A7;font-size:12px">Processing {i+1}/{total}: {f.name}</span>',
                        unsafe_allow_html=True)
                    try:
                        img = Image.open(f).convert("RGBA")
                        res, mask, elapsed = run_pipeline(img, bi_model, bi_tf, bi_device, rm_fn, rm_sess, w_bi)
                        st.session_state.results.append({
                            "name": f.name, "original": img, "result": res,
                            "mask": mask, "time": elapsed, "ok": True, "err": None
                        })
                    except Exception as e:
                        st.session_state.results.append({
                            "name": f.name, "original": None, "result": None,
                            "mask": None, "time": 0, "ok": False, "err": str(e)
                        })
                    prog.progress((i+1)/total)

                status.markdown(
                    f'<span style="color:#00C9A7;font-size:12px;font-weight:600">'
                    f'✓ Done — {total} image(s) processed</span>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            '<div class="info-tag"><b>Step 1</b> — BiRefNet detects fine edges</div>'
            '<div class="info-tag"><b>Step 2</b> — rembg IS-Net wide-domain mask</div>'
            '<div class="info-tag"><b>Step 3</b> — Weighted ensemble merge</div>'
            '<div class="info-tag"><b>Step 4</b> — Guided filter + bilateral refinement</div>'
            '<div class="info-tag"><b>Step 5</b> — Transparent PNG output</div>',
            unsafe_allow_html=True)

    # ── RIGHT: results inline ──────────────────────────────────────────────
    with col_right:
        results = st.session_state.results
        if not results:
            st.markdown(
                '<div style="height:300px;display:flex;align-items:center;justify-content:center;'
                'color:#1C1F29;font-size:13px;border:1px dashed #1C1F29;border-radius:10px;">'
                'Results will appear here</div>',
                unsafe_allow_html=True)
        else:
            # Summary bar
            ok = [r for r in results if r["ok"]]
            fail = [r for r in results if not r["ok"]]
            times = [r["time"] for r in ok]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(results))
            c2.metric("OK", len(ok))
            c3.metric("Failed", len(fail))
            c4.metric("Avg time", f"{np.mean(times):.1f}s" if times else "—")

            st.markdown("---")

            if not ok:
                st.error("All images failed.")
            else:
                # Show ALL results inline — no tab switching needed
                for r in ok:
                    with st.container():
                        name_stem = r["name"].rsplit(".",1)[0]
                        st.markdown(
                            f'<span class="chip-ok">✓ {r["name"]}</span>'
                            f'<span style="color:#404660;font-size:11px;margin-left:8px">'
                            f'{r["time"]:.2f}s</span>',
                            unsafe_allow_html=True)

                        p1, p2, p3 = st.columns(3)
                        with p1:
                            st.markdown('<div class="panel-label">Original</div>', unsafe_allow_html=True)
                            st.image(r["original"].convert("RGB"), use_container_width=True)
                        with p2:
                            st.markdown('<div class="panel-label">Mask</div>', unsafe_allow_html=True)
                            st.image(Image.fromarray(r["mask"]), use_container_width=True)
                        with p3:
                            st.markdown('<div class="panel-label">Result</div>', unsafe_allow_html=True)
                            st.markdown('<div class="checker">', unsafe_allow_html=True)
                            st.image(checker_composite(r["result"]), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        dl_a, dl_b = st.columns(2)
                        with dl_a:
                            st.markdown('<div class="dl-teal">', unsafe_allow_html=True)
                            st.download_button(
                                "↓ Download PNG",
                                data=pil2bytes(r["result"], "PNG"),
                                file_name=f"{name_stem}_nobg.png",
                                mime="image/png",
                                key=f"dl_{r['name']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with dl_b:
                            st.markdown('<div class="dl-teal">', unsafe_allow_html=True)
                            st.download_button(
                                "↓ Download mask",
                                data=pil2bytes(Image.fromarray(r["mask"]), "PNG"),
                                file_name=f"{name_stem}_mask.png",
                                mime="image/png",
                                key=f"dlm_{r['name']}")
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<hr style="margin:12px 0">', unsafe_allow_html=True)

            if fail:
                with st.expander(f"Failed ({len(fail)})"):
                    for r in fail:
                        st.markdown(
                            f'<span class="chip-err">{r["name"]}</span>'
                            f'<span style="color:#64748B;font-size:12px;margin-left:8px">{r["err"]}</span>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 2 — BATCH DOWNLOAD
# ══════════════════════════════════════════════════
with tab_batch:
    ok = [r for r in st.session_state.results if r["ok"]]
    if not ok:
        st.info("Process images first in the Process tab.")
    else:
        st.markdown(f"#### {len(ok)} image(s) ready")
        bc1, bc2 = st.columns(2)

        with bc1:
            z = io.BytesIO()
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in ok:
                    zf.writestr(f"{r['name'].rsplit('.',1)[0]}_nobg.png", pil2bytes(r["result"]))
            z.seek(0)
            st.markdown('<div class="dl-teal">', unsafe_allow_html=True)
            st.download_button(f"↓ All transparent PNGs ({len(ok)})", data=z.getvalue(),
                file_name="results_nobg.zip", mime="application/zip", key="dl_all")
            st.markdown('</div>', unsafe_allow_html=True)

        with bc2:
            z2 = io.BytesIO()
            with zipfile.ZipFile(z2, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in ok:
                    zf.writestr(f"{r['name'].rsplit('.',1)[0]}_mask.png",
                        pil2bytes(Image.fromarray(r["mask"])))
            z2.seek(0)
            st.markdown('<div class="dl-teal">', unsafe_allow_html=True)
            st.download_button(f"↓ All alpha masks ({len(ok)})", data=z2.getvalue(),
                file_name="results_masks.zip", mime="application/zip", key="dl_all_masks")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        cols = st.columns(4)
        for i, r in enumerate(ok):
            with cols[i % 4]:
                st.image(checker_composite(r["result"]),
                    caption=r["name"][:18], use_container_width=True)

st.markdown(
    '<div style="text-align:center;font-size:11px;color:#1C1F29;padding:24px 0 8px;'
    'border-top:1px solid #1C1F29;margin-top:32px">'
    'Background Removal v5.2 · BiRefNet + IS-Net · Ahmed Nasser</div>',
    unsafe_allow_html=True)