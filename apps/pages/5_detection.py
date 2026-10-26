"""Streamlit page for the balloon detection pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


import streamlit as st

from cvkit.detection.pipeline import (
    PROJECT_ROOT,
    StepExecutionError,
    run_evaluation,
    run_full_pipeline,
    run_inference,
    run_proposal_generation,
    run_training,
)
from cvkit.detection.config import resolve_dataset_root


st.set_page_config(page_title="Balloon Detection Pipeline", layout="wide")
st.title("Balloon Detection Pipeline - Selective Search + HOG + SVM")
st.caption("CPU detection pipeline. Outputs are written under outputs/detection/.")


def project_relative(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


dataset_root = resolve_dataset_root("data/balloon_dataset")
test_images = sorted((dataset_root / "test").glob("*.jpg"))
if not test_images:
    st.error(f"No test JPG files found in {dataset_root / 'test'}")
    st.stop()


with st.sidebar:
    st.header("Pipeline Parameters")

    st.subheader("Proposal")
    proposal_split = st.selectbox("Split", ("train", "valid", "test", "all"), index=3)
    proposal_scale = st.number_input("Scale", min_value=1.0, value=500.0, step=50.0)
    proposal_sigma = st.number_input(
        "Sigma", min_value=0.0, value=0.8, step=0.1, format="%.2f"
    )
    proposal_min_size = st.number_input("Min size", min_value=1, value=20, step=1)
    proposal_maximum = st.number_input(
        "Max proposals", min_value=1, value=2000, step=100
    )
    proposal_force = st.checkbox("Force regenerate proposals", value=False)

    st.subheader("Training")
    training_tp = st.number_input(
        "TP positive threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )
    training_tn = st.number_input(
        "TN negative threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    max_positive = st.number_input(
        "Max positive per image", min_value=1, value=32, step=1
    )
    max_negative = st.number_input(
        "Max negative per image", min_value=1, value=96, step=1
    )
    negative_ratio = st.number_input(
        "Negative / positive ratio", min_value=0.1, value=5.0, step=0.5
    )

    st.subheader("Inference")
    selected_image = st.selectbox(
        "Test image", test_images, format_func=lambda image_path: image_path.name
    )
    model_path = st.text_input(
        "Model path", value="outputs/detection/models/balloon_svm.joblib"
    )
    inference_score = st.number_input(
        "Inference score threshold", value=0.5, step=0.1, format="%.3f"
    )
    inference_nms = st.number_input(
        "Inference NMS threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    reuse_proposals = st.checkbox("Reuse cached proposals", value=True)

    st.subheader("Evaluation")
    evaluation_score = st.number_input(
        "Evaluation score threshold", value=0.0, step=0.1, format="%.3f"
    )
    evaluation_nms = st.number_input(
        "Evaluation NMS threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    max_detections = st.number_input(
        "Max detections per image", min_value=1, value=100, step=10
    )


UI_CONFIG = {
    "dataset_root": project_relative(dataset_root),
    "results_root": "outputs/detection",
    "run_proposal_generation": True,
    "run_training": True,
    "run_inference": True,
    "run_evaluation": True,
    "proposal": {
        "split": proposal_split,
        "scale": float(proposal_scale),
        "sigma": float(proposal_sigma),
        "min_size": int(proposal_min_size),
        "max_proposals": int(proposal_maximum),
        "force": proposal_force,
    },
    "training": {
        "tp": float(training_tp),
        "tn": float(training_tn),
        "max_positive": int(max_positive),
        "max_negative": int(max_negative),
        "negative_positive_ratio": float(negative_ratio),
    },
    "inference": {
        "input_image": project_relative(selected_image),
        "model_path": model_path,
        "score_threshold": float(inference_score),
        "nms_threshold": float(inference_nms),
        "reuse_proposals": reuse_proposals,
    },
    "evaluation": {
        "model_path": model_path,
        "score_threshold": float(evaluation_score),
        "nms_threshold": float(evaluation_nms),
        "max_detections": int(max_detections),
    },
}


def execute(label, function):
    try:
        with st.spinner(f"{label} is running..."):
            result = function(UI_CONFIG)
    except StepExecutionError as exc:
        st.error(str(exc))
        if exc.log:
            st.code(exc.log, language="text")
        return None
    except Exception as exc:
        st.error(f"{label} failed: {type(exc).__name__}: {exc}")
        return None
    st.success(f"{label} completed")
    return result


if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}

button_columns = st.columns(5)
generate_clicked = button_columns[0].button("Generate Proposals", width="stretch")
train_clicked = button_columns[1].button("Train SVM", width="stretch")
inference_clicked = button_columns[2].button(
    "Run Inference on Selected Image", width="stretch"
)
evaluation_clicked = button_columns[3].button(
    "Evaluate on Test Set", width="stretch"
)
full_clicked = button_columns[4].button(
    "Run Full Pipeline", type="primary", width="stretch"
)

if generate_clicked:
    result = execute("Proposal generation", run_proposal_generation)
    if result:
        st.session_state.pipeline_results["proposal_generation"] = result
if train_clicked:
    result = execute("SVM training", run_training)
    if result:
        st.session_state.pipeline_results["training"] = result
if inference_clicked:
    result = execute("Inference", run_inference)
    if result:
        st.session_state.pipeline_results["inference"] = result
if evaluation_clicked:
    result = execute("Evaluation", run_evaluation)
    if result:
        st.session_state.pipeline_results["evaluation"] = result
if full_clicked:
    result = execute("Full pipeline", run_full_pipeline)
    if result:
        st.session_state.pipeline_results.update(result)


results = st.session_state.pipeline_results

if "proposal_generation" in results:
    result = results["proposal_generation"]
    st.subheader("Proposal Generation Result")
    rows = [
        {
            "split": split,
            "generated images": summary["generated"],
            "skipped images": summary["skipped"],
            "cached images": summary["cached_images"],
            "proposals": summary["proposals"],
        }
        for split, summary in result["splits"].items()
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.write(f"Proposal output: `{project_relative(result['proposal_root'])}`")
    with st.expander("Proposal log"):
        st.code(result["log"] or "No log output", language="text")

if "training" in results:
    result = results["training"]
    st.subheader("Training Result")
    columns = st.columns(3)
    columns[0].metric("Positive samples", result["positive_samples"])
    columns[1].metric("Negative samples", result["negative_samples"])
    columns[2].metric("Total samples", result["total_samples"])
    st.write(f"Model: `{project_relative(result['model_path'])}`")
    with st.expander("Training log"):
        st.code(result["log"] or "No log output", language="text")

if "inference" in results:
    result = results["inference"]
    st.subheader("Inference Result")
    st.metric("Detections", result["detection_count"])
    st.write(
        f"Detection image: `{project_relative(result['detection_image_path'])}`"
    )
    st.write(f"Detection JSON: `{project_relative(result['detection_json_path'])}`")
    if Path(result["detection_image_path"]).is_file():
        st.image(
            str(result["detection_image_path"]),
            caption="Balloon detections",
            width="stretch",
        )
    with st.expander("Inference log"):
        st.code(result["log"] or "No log output", language="text")

if "evaluation" in results:
    result = results["evaluation"]
    st.subheader("Evaluation Result")
    columns = st.columns(3)
    columns[0].metric("MABO", f"{result['mabo']:.4f}")
    columns[1].metric("mAP@[.50:.95]", f"{result['map']:.4f}")
    columns[2].metric("Backend", result["map_backend"])
    st.write(f"Metrics JSON: `{project_relative(result['metrics_path'])}`")
    st.markdown("**Per-image statistics**")
    st.dataframe(result["per_image"], width="stretch", hide_index=True)
    st.markdown("**metrics.json**")
    st.json(result["metrics"])
    with st.expander("Evaluation log"):
        st.code(result["log"] or "No log output", language="text")

if not results:
    st.info("Choose parameters in the sidebar, then run a pipeline step.")
