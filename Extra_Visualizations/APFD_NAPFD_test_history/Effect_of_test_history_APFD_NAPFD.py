import plotly.graph_objects as go

years = ['Cisco', 'IOFROL', 'Paint Control', 'GSDTSR']

fig = go.Figure()
fig.add_trace(go.Bar(x=years,
                y=[0.34, 0.67, 0.60, 0.88],
                name='APFD on 4 cycles',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=years,
                y=[0.52, 0.67, 0.76, 0.94],
                name='APFD on > 4 cycles',
                marker_color='rgb(26, 118, 255)'
                ))
fig.add_trace(go.Bar(x=years,
                y=[0.51, 0.48, 0.10, 0.093],
                name='NAPFD on 4 cycles',
    marker_color='indianred'
                ))
fig.add_trace(go.Bar(x=years,
                y=[0.520, 0.56, 0.52, 0.79],
                name='NAPFD on > 4 cycles',
    marker_color='lightsalmon'
                ))

fig.update_layout(
    #title='US Export of Plastic Scrap',
    xaxis_tickfont_size=14,
    
    yaxis=dict(
        title='APFD / NAFPD',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.6, # gap between bars of adjacent location coordinates.
    bargroupgap=0.15, # gap between bars of the same location coordinate.
    width=650, height=450
)
#fig.update_xaxes(automargin=True)
fig.show()
#fig.write_image("APFD_NAPFD.pdf")
