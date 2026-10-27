from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DATA_RAW, MODELS_DIR, SENSOR_DESCRIPTIONS, SUBSETS  # noqa: E402
from src.inference import (DEFAULT_CRITICAL, DEFAULT_WARNING, fleet_status, load_artifacts,  # noqa: E402
                           predict_engine_curve, rul_status, true_rul_curve)
from src.ingestion import load_subset  # noqa: E402

st.set_page_config(page_title="Estimación de RUL — Turbofan", page_icon="✈️", layout="wide")

STATUS_COLORS = {"NORMAL": "#2e7d32", "PRECAUCIÓN": "#f9a825", "CRÍTICO": "#c62828"}


def get_feature_description(feature: str) -> str:
    """Obtiene de forma segura la descripción de un sensor base o derivado."""
    if feature in SENSOR_DESCRIPTIONS:
        return SENSOR_DESCRIPTIONS[feature]
    for suffix, label in [("_roll_mean", "Media Móvil"), ("_roll_std", "Desviación Estándar Móvil")]:
        if feature.endswith(suffix):
            base = feature[:-len(suffix)]
            base_desc = SENSOR_DESCRIPTIONS.get(base, base)
            return f"{label} de {base} ({base_desc})"
    return feature


# carga
@st.cache_resource(show_spinner="Cargando modelo…")
def get_artifacts(subset: str):
    return load_artifacts(subset, MODELS_DIR)


@st.cache_data(show_spinner="Cargando datos…")
def get_data(subset: str):
    d = load_subset(subset, DATA_RAW)
    return d.test, d.rul_test


def available_subsets() -> list[str]:
    return [s for s in SUBSETS if (MODELS_DIR / s / "model.keras").exists()]


# gráficos
def gauge(rul: float, cap: float, warning: float, critical: float, status: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(rul),
        number={"suffix": " ciclos", "font": {"size": 34}},
        title={"text": "RUL estimada (último ciclo)"},
        gauge={
            "axis": {"range": [0, cap]},
            "bar": {"color": STATUS_COLORS[status]},
            "steps": [
                {"range": [0, critical], "color": "#ffcdd2"},
                {"range": [critical, warning], "color": "#fff3c4"},
                {"range": [warning, cap], "color": "#c8e6c9"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": critical},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def sensor_chart(df: pd.DataFrame, sensors: list[str]) -> go.Figure:
    long = df.melt(id_vars="cycle", value_vars=sensors, var_name="sensor", value_name="valor")
    
    def format_sensor_name(s):
        desc = get_feature_description(s)
        short_desc = desc.split(" – ")[0] if " – " in desc else desc.split(" (")[0]
        return f"{s} · {short_desc}"

    long["sensor"] = long["sensor"].map(format_sensor_name)
    fig = px.line(long, x="cycle", y="valor", facet_row="sensor", facet_row_spacing=0.03)
    fig.update_yaxes(matches=None, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=0, x=0.01,
                                                xanchor="left", yanchor="bottom"))
    fig.update_layout(height=max(260, 170 * len(sensors)), showlegend=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    fig.update_xaxes(title_text="Ciclo de operación")
    return fig


def rul_chart(curve: pd.DataFrame, true_curve, warning: float, critical: float, cap: float,
              window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=critical, fillcolor="#ffcdd2", opacity=0.45, line_width=0)
    fig.add_hrect(y0=critical, y1=warning, fillcolor="#fff3c4", opacity=0.45, line_width=0)
    fig.add_trace(go.Scatter(x=curve["cycle"], y=curve["rul_pred"], mode="lines",
                           name="RUL estimada", line=dict(color="#d62728", width=2.5)))
    if true_curve is not None:
        fig.add_trace(go.Scatter(x=curve["cycle"], y=true_curve, mode="lines",
                               name="RUL real", line=dict(color="#1f77b4", dash="dash", width=2)))
    fig.add_vrect(x0=curve["cycle"].min(), x1=curve["cycle"].min() + window - 1,
                  fillcolor="gray", opacity=0.12, line_width=0,
                  annotation_text="ventana incompleta", annotation_position="top left")
    fig.update_layout(height=420, xaxis_title="Ciclo de operación", yaxis_title="RUL (ciclos)",
                      yaxis_range=[0, cap * 1.05], legend=dict(orientation="h", y=1.08),
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_alert(status: str, rul: float, warning: float, critical: float) -> None:
    if status == "CRÍTICO":
        st.error(f"🚨 **ALERTA DE FALLA — CRÍTICO.** RUL estimada = {rul:.0f} ciclos "
                 f"(≤ {critical:.0f}). Programar intervención en la próxima escala.")
    elif status == "PRECAUCIÓN":
        st.warning(f"⚠️ **PRECAUCIÓN.** RUL estimada = {rul:.0f} ciclos (≤ {warning:.0f}). "
                   "Planificar mantenimiento.")
    else:
        st.success(f"**NORMAL.** RUL estimada = {rul:.0f} ciclos (> {warning:.0f}).")


# app
def main() -> None:
    st.title("Estimación de vida útil restante (RUL) en motores turbofan")
    st.caption("Trabajo Terminal 2027-A190 · ESCOM-IPN · LSTM entrenada sobre NASA C-MAPSS")
    st.info(
        "Los datos son de **simulación** (C-MAPSS) de un motor turbofan genérico; no son "
        "telemetría real de motores Boeing 777. Los umbrales de alerta son parámetros de "
        "política de mantenimiento, ajustables en la barra lateral.",
        icon="ℹ️",
    )

    subsets = available_subsets()
    if not subsets:
        st.error(f"No se encontraron modelos entrenados en `{MODELS_DIR}`. "
                 "Ejecute `python -m src.train` y recargue esta página.")
        st.stop()

    with st.sidebar:
        st.header("Configuración")
        subset = st.selectbox("Subconjunto C-MAPSS", subsets)
        art = get_artifacts(subset)
        try:
            test_df, rul_test = get_data(subset)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        cap = float(art.config["rul_cap"])
        st.subheader("Umbrales de alerta (ciclos)")
        critical = st.slider("Crítico (≤)", 5, int(cap) - 10, DEFAULT_CRITICAL)
        warning = st.slider("Precaución (≤)", critical + 1, int(cap), max(DEFAULT_WARNING, critical + 1))

    units = sorted(test_df["unit"].unique())
    tab_motor, tab_flota, tab_modelo = st.tabs(["🔧 Motor individual", "🛫 Flota (prueba)", "📊 Modelo"])

    # motor individual
    with tab_motor:
        unit = st.selectbox("Motor", units, key="unit")
        eng = test_df[test_df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        rul_final = float(rul_test[units.index(unit)])
        curve = predict_engine_curve(art, eng)
        current = float(curve["rul_pred"].iloc[-1])
        status = rul_status(current, warning, critical)

        render_alert(status, current, warning, critical)
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        c1.plotly_chart(gauge(current, cap, warning, critical, status), width="stretch")
        c2.metric("Ciclos observados", f"{len(eng)}")
        c3.metric("RUL estimada", f"{current:.0f} ciclos")
        c4.metric("RUL real (referencia)", f"{rul_final:.0f} ciclos",
                  delta=f"{current - rul_final:+.0f} (error)", delta_color="off")

        st.subheader("Curva de RUL: estimada vs. real")
        true_c = true_rul_curve(len(eng), rul_final, cap)
        st.plotly_chart(rul_chart(curve, true_c, warning, critical, cap, art.config["window"]),
                        width="stretch")
        st.caption("RUL real = RUL_FD00X.txt + ciclos restantes de la trayectoria truncada, "
                   f"limitada a {cap:.0f} (RUL lineal por tramos).")

        st.subheader("Evolución temporal de los sensores")
        feats = art.preprocessor.feature_cols_
        default = feats[: min(4, len(feats))]
        chosen = st.multiselect(
            "Sensores", 
            feats, 
            default=default,
            format_func=lambda s: f"{s} — {get_feature_description(s)}"
        )
        normalized = st.toggle("Mostrar valores normalizados (por régimen)", value=False)
        if chosen:
            if normalized:
                plot_df = art.preprocessor.transform(eng)[["cycle", *chosen]]
            else:
                plot_df = eng[["cycle", *chosen]]
            st.plotly_chart(sensor_chart(plot_df, chosen), width="stretch")
        else:
            st.info("Seleccione al menos un sensor.")

    # flota
    with tab_flota:
        st.subheader("Priorización de intervenciones")
        fleet = fleet_status(art, test_df, warning, critical)
        counts = fleet["estado"].value_counts()
        k1, k2, k3 = st.columns(3)
        k1.metric("🚨 Críticos", int(counts.get("CRÍTICO", 0)))
        k2.metric("⚠️ Precaución", int(counts.get("PRECAUCIÓN", 0)))
        k3.metric("✅ Normales", int(counts.get("NORMAL", 0)))
        fig = px.bar(fleet, x="motor", y="rul_estimada", color="estado",
                     color_discrete_map=STATUS_COLORS, category_orders={"motor": fleet["motor"].tolist()},
                     labels={"rul_estimada": "RUL estimada (ciclos)", "motor": "Motor"})
        fig.add_hline(y=critical, line_dash="dot", line_color="#c62828")
        fig.add_hline(y=warning, line_dash="dot", line_color="#f9a825")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(fleet, width="stretch", hide_index=True)

    # modelo
    with tab_modelo:
        st.subheader(f"Modelo {subset}")
        if art.metrics:
            m = art.metrics["test"]
            a, b, c = st.columns(3)
            a.metric("RMSE (prueba)", f"{m['rmse']:.2f}")
            b.metric("MAE (prueba)", f"{m['mae']:.2f}")
            c.metric("Score (prueba)", f"{m['score']:.0f}")
            st.caption("Métricas sobre el conjunto de prueba contra RUL real sin tope. "
                       "Score menor es mejor; penaliza más las estimaciones tardías (RUL sobreestimada).")
            st.write("**Sensores utilizados:**", ", ".join(art.metrics.get("features", [])))
            st.write("**Sensores descartados:**", ", ".join(art.metrics.get("dropped_sensors", [])) or "—")
        st.write("**Hiperparámetros:**")
        st.json(art.config)
        fig_path = MODELS_DIR / subset / "figures" / "pred_vs_real.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="RUL real vs. estimada (conjunto de prueba)")

main()