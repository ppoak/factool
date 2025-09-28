import os
import threading
import time
import logging
from pathlib import Path
import streamlit as st
import pandas as pd

from agent import factool_agent

# Constants
DEFINITIONS_DIR = Path("docs/definitions")
UPLOAD_DIR = Path("tmp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Factool Agent UI", layout="wide")
st.title("Factool Agent — 因子代码生成器")

with st.sidebar:
    st.header("选择或上传说明文档 (Markdown)")

    # List available definition files
    md_files = []
    if DEFINITIONS_DIR.exists():
        md_files = sorted(
            [p for p in DEFINITIONS_DIR.iterdir() if p.suffix.lower() in (".md",)]
        )
    md_options = []
    cwd_resolved = Path.cwd().resolve()
    for p in md_files:
        try:
            rel = p.resolve().relative_to(cwd_resolved)
            md_options.append(rel.as_posix())
        except Exception:
            # fallback to absolute path if relative conversion fails
            md_options.append(p.resolve().as_posix())
    md_options.insert(0, "-- 选择本地文件 --")

    selected = st.selectbox("从 docs/definitions 选择已有说明", md_options)

    uploaded_file = st.file_uploader(
        "或上传你自己的说明 Markdown 文件", type=["md", "markdown"]
    )

    st.markdown("---")
    st.header("运行参数")
    factorpy = st.text_input("输出因子脚本目录 (factorpy)", value="")
    begin = st.text_input("开始时间 (begin)", value="2015-01-01")
    end = st.text_input("结束时间 (end)", value="now")
    save = st.checkbox("是否计算并保存因子数据 (save)", value=False)
    evaluation = st.checkbox("是否进行因子评估 (evaluation)", value=False)

    st.markdown("---")
    st.write("注意: 运行可能需要较长时间，系统会在后台执行并显示日志。")

# Main area: display selected doc content
st.subheader("说明文档预览与编辑")
selected_doc_path = None
file_content = ""

if uploaded_file is not None:
    # Save uploaded file to temp dir
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    selected_doc_path = save_path
    st.info(f"已上传并使用: {save_path}")
    try:
        file_content = save_path.read_text(encoding="utf-8")
    except Exception:
        st.write("无法读取上传文件的文本内容。")
else:
    if selected and selected != "-- 选择本地文件 --":
        # selected is a posix-style string (relative or absolute). Resolve safely to an absolute Path.
        sel_path = Path(selected)
        if not sel_path.is_absolute():
            sel_path = Path.cwd() / sel_path
        selected_doc_path = sel_path.resolve()
        try:
            file_content = selected_doc_path.read_text(encoding="utf-8")
        except Exception:
            st.write("无法读取所选文件的内容。")
    else:
        st.write("请在侧边栏选择一个已有说明文件或上传一个新的 Markdown 文件。")

# Initialize session state for editor and selected path
# Normalize selected path into a canonical resolved string to avoid needless resets
sel_path_str = None
if selected_doc_path is not None:
    try:
        sel_path_str = Path(selected_doc_path).resolve().as_posix()
    except Exception:
        sel_path_str = Path(selected_doc_path).absolute().as_posix()

if sel_path_str:
    # when switching to a new file, reset editing state and editor content
    if st.session_state.get("selected_doc_path") != sel_path_str:
        st.session_state["selected_doc_path"] = sel_path_str
        st.session_state["md_editor"] = file_content
        st.session_state["editing"] = False

# Single-box toggle: preview by default, edit when requested
if st.session_state.get("selected_doc_path"):
    path = Path(st.session_state.get("selected_doc_path"))
    st.write(f"当前文件: {path}")

    def _enter_edit():
        st.session_state["editing"] = True
        try:
            disk = path.read_text(encoding="utf-8")
            st.session_state["md_editor"] = disk
        except Exception:
            pass

    def _save_and_reload():
        if not st.session_state.get("editing"):
            st.session_state["md_editor"] = path.read_text(encoding="utf-8")
            return
        try:
            path.write_text(st.session_state.get("md_editor", ""), encoding="utf-8")
            st.session_state["editing"] = False
            st.session_state["_last_save"] = time.time()
            # Trigger a full page reload by updating query params (works across Streamlit versions)
            try:
                st.query_params(_refresh=int(time.time()))
            except Exception:
                # fallback: do nothing, Streamlit will still rerun after the interaction
                pass
        except Exception as e:
            st.error(f"保存失败: {e}")

    # Render the appropriate button for current state
    col_left, col_right = st.columns(2)
    with col_left:
        if st.button("编辑", key="btn_edit"):
            _enter_edit()
    with col_right:
        if st.button("保存并退出", key="btn_save"):
            _save_and_reload()

    st.markdown("---")

    if st.session_state.get("editing"):
        # Editing mode: show a single text_area for editing
        st.text_area(
            "编辑 Markdown",
            value=st.session_state.get("md_editor", ""),
            key="md_editor",
            height=500,
        )
    else:
        try:
            st.markdown(st.session_state.get("md_editor", ""))
        except Exception:
            st.write("无法渲染 Markdown 预览")

else:
    # no file selected
    pass

# Run button and logs
run_button = st.button("运行生成")

log_box = st.empty()


class ListHandler(logging.Handler):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def emit(self, record):
        msg = self.format(record)
        self.store.append(msg)


def run_generate(doc_path, factorpy, begin, end, save, evaluation, log_store):
    # Attach handler to root logger to capture logs
    handler = ListHandler(log_store)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        factool_agent.generate(
            str(doc_path),
            factorpy or None,
            begin,
            end,
            save,
            evaluation,
        )
        log_store.append("RUN_FINISHED: 生成任务已完成。")
    except Exception as e:
        logging.exception("运行过程中出现错误")
        log_store.append(f"ERROR: {e}")
    finally:
        # remove handler
        root_logger.removeHandler(handler)


if selected_doc_path is not None:
    artifact = Path(os.getenv("EVAL_PATH")) / selected_doc_path.stem
    if artifact.exists():
        for atf in artifact.iterdir():
            st.header(f"Evaluation result {atf.stem}")
            for filename in ["ic.png", "values.png"]:
                st.image(atf / filename)
            df = pd.read_excel(atf / "evaluation.xlsx", sheet_name="TopK and NGroup")
            st.dataframe(df)


if run_button:
    if selected_doc_path is None:
        st.error("未指定说明文档，请选择或上传一个 Markdown 文件。")
    else:
        status = st.empty()
        log_store = []
        thread = threading.Thread(
            target=run_generate,
            args=(selected_doc_path, factorpy, begin, end, save, evaluation, log_store),
            daemon=True,
        )
        thread.start()

        with st.spinner("因子生成中，正在运行... 请耐心等待"):
            while thread.is_alive():
                # update logs in a single placeholder (avoid duplicate widget ids)
                log_text = "\n".join(log_store) if log_store else "(等待日志输出...)"
                log_box.code(log_text, language="text")
                time.sleep(0.5)
        # Final logs
        log_box.code("\n".join(log_store), language="text")
        st.success("任务完成。请查看日志与输出文件。")

        # Offer to show generated file if exists
        doc_stem = Path(selected_doc_path).stem
        output_dir = Path(factorpy or "out")
        output_path = output_dir / f"{doc_stem}.py"
        if output_path.exists():
            st.subheader("生成的因子脚本")
            try:
                st.code(output_path.read_text(encoding="utf-8"), language="python")
                st.write(f"已保存至: {output_path}")
            except Exception:
                st.write(f"生成的脚本位于: {output_path} (无法预览文件内容)")
        else:
            st.write(
                "未在预期位置找到生成的脚本，可能发生错误。请查看日志以获取更多信息。"
            )
