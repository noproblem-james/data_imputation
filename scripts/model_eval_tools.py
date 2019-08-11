
import pandas as pd
import altair as alt

def draw_feature_importance_tornado(feat_imp_df):

    feat_imp_bars = alt.Chart(feat_imp_df, width=1000, height=500, title="Feature Importances").mark_bar().encode(
        alt.X('importance:Q', title="Importance"),
        alt.Y('feature:N', title="Feature", sort=alt.EncodingSortField(field="importance", op="sum", order='descending')),
        alt.Color('importance:Q', legend=None, scale=alt.Scale(type="log", range=["lightskyblue", "steelblue"])),
    )

    text = feat_imp_bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='importance:Q'
    )

    return feat_imp_bars.configure_axis(labelFontSize=12, titleFontSize=20, labelFlush=False, labelLimit=40).configure_title(fontSize=30)

def draw_resid_dashboard(resids_df, target_units, cat_col, tooltip_features):

    brush = alt.selection(type='interval', resolve='global')

    max_act = resids_df["actual"].max() * 1.1
    max_pred = resids_df["pred"].max()* 1.1
    max_pred_act = max(max_act, max_pred)
    max_abs_resid = resids_df["abs_resid"].max()* 1.1
    print(max_pred_act)

    ## Scatterplots

    scatter_base = \
    alt.Chart().mark_circle(size=70, clip=True).encode(
        x=alt.X("actual:Q", title="Actual Value", scale=alt.Scale(domain=(0, max_pred_act))),
        color=alt.condition(brush, cat_col, alt.ColorValue('gray'), legend=None),
        tooltip=tooltip_features
    ).add_selection(brush
    ).properties(
        width=610,
        height=610
    )

    # Actual vs Predicted Scatterplot
    pred_scatter = \
    scatter_base.encode(y=alt.Y("pred:Q", title="Predicted Value", scale=alt.Scale(domain=(0, max_pred_act)))
                       ).properties(title=f"Actuals vs. Predictions, colored by {cat_col}")

    identity_rule_df = pd.DataFrame([{"x": 0, "y": 0},  {"x": max_pred_act, "y": max_pred_act}])
    identity_rule = alt.Chart(identity_rule_df).mark_line(color="#000000", clip=True).encode(x=alt.X("x:Q", scale=alt.Scale(domain=(0, max_pred_act))),y='y:Q',)

    pred_scatter += identity_rule # Draw the line on top

    # Actual vs Residual
    resid_scatter = \
    scatter_base.encode(y=alt.Y("resid:Q", title="Residual (Predicted minus Actual)", scale=alt.Scale(domain=(-max_abs_resid, max_abs_resid)))
        ).properties(title=f"Actuals vs. Residuals, colored by {cat_col}")#.interactive()

    zero_rule_df = pd.DataFrame([{"x": 0, "y": 0}, {"x": max_pred_act, "y": 0}])
    zero_rule = alt.Chart(zero_rule_df).mark_line(color="#000000").encode(x=alt.X("x:Q", scale=alt.Scale(domain=(0, max_pred_act))),y='y:Q',)

    resid_scatter += zero_rule # Draw the line on top

    scatterplots = pred_scatter | resid_scatter

    ## Bar Charts

    bars_base = \
    alt.Chart().mark_bar().encode(
        y=alt.Y(f'{cat_col}:N', title=None),
        color=alt.Color(f'{cat_col}:N', legend=None)
    ).properties(
        width=350,
        height=150
    ).transform_filter(brush)


    vc_bars = bars_base.encode(x='count({cat_col}):Q', y=alt.Y(f'{cat_col}:N')).properties(title="Value Counts")
    mape_bars = bars_base.encode(x=alt.X('mean(abs_perc_resid):Q', title="Percent")).properties(title="Mean Absolute Percent Error (MAPE)")

    #Stacked bars
    avg_act_bars = bars_base.encode(alt.X('mean(actual):Q', stack=None))
    mae_bars = bars_base.encode(alt.X('mean(abs_resid):Q', title=target_units, stack=None),
                           color=alt.value("black")
                          ).properties(title="Mean Absolute Error (black) vs. Mean of Target")
    avg_act_vs_mae = avg_act_bars + mae_bars

    bar_charts = alt.HConcatChart(hconcat=(vc_bars, avg_act_vs_mae, mape_bars),)

    dash = alt.VConcatChart(data=resids_df, vconcat=(scatterplots, bar_charts))

    return dash.configure_axis(labelFontSize=15, titleFontSize=15, labelFlush=True, labelLimit=400).configure_title(fontSize=20)
