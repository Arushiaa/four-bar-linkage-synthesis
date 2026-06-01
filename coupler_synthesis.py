"""
AI Assistance Declaration:

Some parts of this project were developed with the help of AI tools (ChatGPT, Claude).
AI was used for:
- Debugging and fixing errors (Streamlit, NumPy issues)
- Improving UI layout and input handling

The core concepts, mechanism synthesis logic, and overall design were understood,
implemented, and verified independently.
"""

"""
4-Point Coupler Curve Synthesis
================================
Tab 1 — Numerical  (exact, scipy least-squares)
Tab 2 — Graphical  (Burmester construction, educational)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares

st.set_page_config(layout="wide", page_title="4-Point Coupler Synthesis")

# ══════════════════════════════════════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])

def circle_isect_both(c1, r1, c2, r2):
    dv = c2 - c1; d = np.linalg.norm(dv)
    if d < 1e-12 or d > r1+r2+1e-9 or d < abs(r1-r2)-1e-9: return []
    a = (r1*r1-r2*r2+d*d)/(2*d); h2 = r1*r1-a*a
    if h2 < -1e-9: return []
    h = np.sqrt(max(h2, 0)); u = dv/d; p = c1+a*u; tv = np.array([-u[1], u[0]])
    return [p+h*tv, p-h*tv] if h > 1e-9 else [p]

def circle_isect_one(c1, r1, c2, r2, branch=0):
    pts = circle_isect_both(c1, r1, c2, r2)
    return pts[branch % len(pts)] if pts else None

def perp_bisector(p1, p2):
    mid = (p1+p2)/2.0; d = p2-p1
    return mid, np.array([-d[1], d[0]])/np.linalg.norm(d)

def circumcenter(a, b, c):
    D = 2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
    if abs(D) < 1e-9: return None
    na, nb, nc = a@a, b@b, c@c
    return np.array([(na*(b[1]-c[1])+nb*(c[1]-a[1])+nc*(a[1]-b[1]))/D,
                     (na*(c[0]-b[0])+nb*(a[0]-c[0])+nc*(b[0]-a[0]))/D])

# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════
def num_residuals(x, targets):
    phis=x[0:4]; r=abs(x[4]); HC=np.array(x[5:7]); O4=np.array(x[7:9]); Bb=np.array(x[9:11])
    if r < 0.1 or np.linalg.norm(Bb) < 0.1: return np.ones(6)*1e3
    As = [targets[i]-r*np.array([np.cos(phis[i]),np.sin(phis[i])]) for i in range(4)]
    Bs = [As[i]+rot(phis[i])@Bb for i in range(4)]
    dA0=np.linalg.norm(As[0]-HC); dB0=np.linalg.norm(Bs[0]-O4)
    if dA0 < 0.1 or dB0 < 0.1: return np.ones(6)*1e3
    res = []
    for i in range(1, 4):
        res.append(np.linalg.norm(As[i]-HC)-dA0)
        res.append(np.linalg.norm(Bs[i]-O4)-dB0)
    return np.array(res)

def simulate_trace(HC, O4, r_crank, r_rocker, r_AB, phi_AB0, C_body, A1, B1_init, n=360):
    theta0 = np.arctan2((A1-HC)[1], (A1-HC)[0])
    prev_B = B1_init.copy(); trace = []
    for theta in np.linspace(theta0, theta0+2*np.pi, n, endpoint=False):
        A = HC+r_crank*np.array([np.cos(theta), np.sin(theta)])
        opts = circle_isect_both(O4, r_rocker, A, r_AB)
        if not opts: continue
        B = min(opts, key=lambda o: np.linalg.norm(o-prev_B)); prev_B = B.copy()
        trace.append(A+rot(np.arctan2((B-A)[1],(B-A)[0])-phi_AB0)@C_body)
    return np.array(trace) if trace else np.zeros((0,2))

def num_solve(targets, n_trials=400, seed=0, progress_cb=None):
    targets = [np.array(t, dtype=float) for t in targets]
    pts = np.array(targets)
    scale = max(np.ptp(pts[:,0]), np.ptp(pts[:,1]))*0.8+2.0
    rng = np.random.default_rng(seed); good = []

    for trial in range(n_trials):
        sc = rng.uniform(4, max(scale*1.5, 10))
        x0 = ([rng.uniform(-np.pi, np.pi) for _ in range(4)] +
               [rng.uniform(0.5, sc*0.6)] +
               [rng.uniform(-sc, sc), rng.uniform(-sc, sc)] +
               [rng.uniform(-sc, sc), rng.uniform(-sc, sc)] +
               [rng.uniform(-sc*0.5, sc*0.5), rng.uniform(-sc*0.5, sc*0.5)])
        try:
            sol = least_squares(num_residuals, x0, args=(targets,), method='trf',
                                ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=1500)
            if sol.cost > 1e-8: continue
            x=sol.x; phis=x[0:4]; r=abs(x[4])
            HC=np.array(x[5:7]); O4=np.array(x[7:9]); Bb=np.array(x[9:11])
            if r < 0.3 or np.linalg.norm(Bb) < 0.3: continue
            As=[targets[i]-r*np.array([np.cos(phis[i]),np.sin(phis[i])]) for i in range(4)]
            Bs=[As[i]+rot(phis[i])@Bb for i in range(4)]
            r_crank=np.linalg.norm(As[0]-HC); r_rocker=np.linalg.norm(Bs[0]-O4)
            r_AB=np.linalg.norm(Bb); ground=np.linalg.norm(HC-O4)
            phi_AB0=np.arctan2(Bb[1],Bb[0]); C_body=np.array([r,0.0])
            opts0=circle_isect_both(O4,r_rocker,As[0],r_AB)
            if not opts0: continue
            # Try both branches — critical for correct trace
            best_err=1e9; best_B1=None
            for B1_try in opts0:
                tr=simulate_trace(HC,O4,r_crank,r_rocker,r_AB,phi_AB0,C_body,As[0],B1_try)
                if len(tr)<340: continue
                err=max(np.linalg.norm(tr-T,axis=1).min() for T in targets)
                if err<best_err: best_err=err; best_B1=B1_try
            if best_err>0.05 or best_B1 is None: continue
            good.append(dict(HC=HC,O4=O4,As=As,Bs=Bs,r=r,
                             r_crank=r_crank,r_rocker=r_rocker,r_AB=r_AB,ground=ground,
                             phi_AB0=phi_AB0,C_body=C_body,B1_init=best_B1,max_err=best_err))
        except Exception:
            pass
        if progress_cb: progress_cb((trial+1)/n_trials)
        if len(good) >= 3: break

    good.sort(key=lambda p: p['max_err'])
    return good

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHICAL (BURMESTER) CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def graphical_construct(targets, hr_t, R_val, r_val, hc_t, bA1, bA3, bA2, bA4):
    Cs = [np.array(t, dtype=float) for t in targets]
    C1,C2,C3,C4 = Cs
    try:
        mid13,p13=perp_bisector(C1,C3); HR=mid13+hr_t*p13
        A1=circle_isect_one(HR,R_val,C1,r_val,bA1)
        A3=circle_isect_one(HR,R_val,C3,r_val,bA3)
        if A1 is None or A3 is None:
            return None, "A1/A3 not found — circles (HR,R) and (Ci,r) don't overlap. Try increasing R."
        midA,pA=perp_bisector(A1,A3); HC=midA+hc_t*pA
        rc=np.linalg.norm(A1-HC)
        if rc<0.1: return None, "Degenerate: HC too close to midpoint of A1─A3."
        A2=circle_isect_one(HC,rc,C2,r_val,bA2)
        A4=circle_isect_one(HC,rc,C4,r_val,bA4)
        if A2 is None: return None, "A2 not found — crank circle and (C2,r) don't intersect."
        if A4 is None: return None, "A4 not found — crank circle and (C4,r) don't intersect."
        As=[A1,A2,A3,A4]
        phi=[np.arctan2((Cs[i]-As[i])[1],(Cs[i]-As[i])[0]) for i in range(4)]
        P=[A1+rot(phi[0]-phi[i])@(HR-As[i]) for i in range(4)]
        B1=circumcenter(P[0],P[1],P[2])
        if B1 is None: return None, "P1,P2,P3 collinear — degenerate. Try different slider values."
        if np.linalg.norm(B1-A1)<0.1: return None, "Degenerate: B1 coincides with A1."
        Bs=[As[i]+rot(phi[i]-phi[0])@(B1-A1) for i in range(4)]
        O4=circumcenter(Bs[0],Bs[1],Bs[2])
        if O4 is None: return None, "B1,B2,B3 collinear — O4 undefined."
        rr=np.linalg.norm(Bs[0]-O4)
        return dict(Cs=Cs,As=As,Bs=Bs,HR=HR,HC=HC,O4=O4,P=P,
                    r_crank=rc,r_val=r_val,r_rocker=rr,
                    r_AB=np.linalg.norm(B1-A1),
                    closure_err=abs(np.linalg.norm(Bs[3]-O4)-rr)), None
    except Exception as e:
        return None, str(e)

def auto_find_graphical(targets):
    """Search for slider values that give low closure error. Returns (hr,R,r,hc,branches)."""
    Cs=[np.array(t,dtype=float) for t in targets]
    pts=np.array(Cs); scale=max(np.ptp(pts[:,0]),np.ptp(pts[:,1]))+2
    rng=np.random.default_rng(0)
    best_err=1e9; best=(3.0,8.0,3.0,2.0,(0,0,0,0))
    for _ in range(5000):
        hr=rng.uniform(-scale,scale); R=rng.uniform(scale*0.2,scale*1.5)
        r=rng.uniform(0.3,scale*0.6); hc=rng.uniform(-scale,scale)
        for br in [(0,0,0,0),(0,1,0,0),(1,0,0,0),(1,1,0,0),
                   (0,0,1,0),(0,1,1,0),(1,0,1,0),(0,1,1,1),(1,1,1,1)]:
            res,_=graphical_construct(Cs,hr,R,r,hc,*br)
            if res and res['closure_err']<best_err:
                best_err=res['closure_err']; best=(hr,R,r,hc,br)
    return best, best_err

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════════════
def animate_frames(p, targets, n_frames=360, speed_ms=20):
    HC,O4=p['HC'],p['O4']
    rc,rr,rab=p['r_crank'],p['r_rocker'],p['r_AB']
    phi0,Cb=p['phi_AB0'],p['C_body']
    theta0=np.arctan2((p['As'][0]-HC)[1],(p['As'][0]-HC)[0])
    prev_B=p['B1_init'].copy(); frames=[]; tx=[]; ty=[]
    for theta in np.linspace(theta0,theta0+2*np.pi,n_frames,endpoint=False):
        A=HC+rc*np.array([np.cos(theta),np.sin(theta)])
        opts=circle_isect_both(O4,rr,A,rab)
        if not opts: continue
        B=min(opts,key=lambda o:np.linalg.norm(o-prev_B)); prev_B=B.copy()
        C=A+rot(np.arctan2((B-A)[1],(B-A)[0])-phi0)@Cb
        tx.append(float(C[0])); ty.append(float(C[1]))
        frames.append(go.Frame(name=str(len(frames)), data=[
            go.Scatter(x=[HC[0],A[0]],y=[HC[1],A[1]],mode='lines',
                       line=dict(color='#1D9E75',width=4),name='Crank'),
            go.Scatter(x=[A[0],C[0],B[0],A[0]],y=[A[1],C[1],B[1],A[1]],
                       mode='lines',fill='toself',fillcolor='rgba(127,119,221,0.15)',
                       line=dict(color='#7F77DD',width=4),name='Coupler'),
            go.Scatter(x=[B[0],O4[0]],y=[B[1],O4[1]],mode='lines',
                       line=dict(color='#378ADD',width=4),name='Rocker'),
            go.Scatter(x=list(tx),y=list(ty),mode='lines',
                       line=dict(color='rgba(239,159,39,0.9)',width=2),name='Trace'),
            go.Scatter(x=[t[0] for t in targets],y=[t[1] for t in targets],
                       mode='markers+text',text=['C1','C2','C3','C4'],
                       textposition='top center',
                       marker=dict(size=14,color='#E24B4A',symbol='x',line=dict(width=3)),
                       name='Precision pts'),
            go.Scatter(x=[A[0],B[0],C[0]],y=[A[1],B[1],C[1]],
                       mode='markers+text',text=['A','B','C'],textposition='top center',
                       marker=dict(size=9,color=['#1D9E75','#378ADD','#E24B4A']),
                       showlegend=False),
        ]))
    return frames, np.array(tx), np.array(ty)

def animate_graphical(gres, targets, n_frames=240):
    # Convert graphical result → animation-compatible dict
    p = {
        'HC': gres['HC'],
        'O4': gres['O4'],
        'As': gres['As'],
        'Bs': gres['Bs'],
        'r_crank': gres['r_crank'],
        'r_rocker': gres['r_rocker'],
        'r_AB': gres['r_AB'],
        'phi_AB0': np.arctan2(
            (gres['Bs'][0]-gres['As'][0])[1],
            (gres['Bs'][0]-gres['As'][0])[0]
        ),
        'C_body': np.array([gres['r_val'], 0.0]),
        'B1_init': gres['Bs'][0]
    }

    frames, tx, ty = animate_frames(p, targets, n_frames=n_frames)

    if not frames:
        return None

    xc, yc, span = compute_range([targets, gres['As'], gres['Bs'], [gres['HC']], [gres['O4']]])

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            **axis_layout(xc, yc, span, h=650),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1.1,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=20, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate')]),
                ])],
            sliders=[dict(
                steps=[dict(method='animate',
                            args=[[f.name], dict(mode='immediate',
                                                 frame=dict(duration=0, redraw=True))],
                            label='')
                       for f in frames],
                transition=dict(duration=0),
                x=0.05,
                y=0,
                len=0.9,
                currentvalue=dict(visible=False)
            )]
        )
    )

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def axis_layout(xc, yc, span, h=580):
    return dict(height=h,
        xaxis=dict(range=[xc-span,xc+span],scaleanchor='y',scaleratio=1,
                   zeroline=True,zerolinewidth=1),
        yaxis=dict(range=[yc-span,yc+span],zeroline=True,zerolinewidth=1),
        legend=dict(orientation='h',y=-0.1),
        margin=dict(l=40,r=20,t=20,b=60))

def compute_range(point_lists):
    pts=np.vstack(point_lists)
    pad=max(np.ptp(pts[:,0]),np.ptp(pts[:,1]))*0.3+3
    xc,yc=pts[:,0].mean(),pts[:,1].mean()
    return xc,yc,max(np.ptp(pts[:,0]),np.ptp(pts[:,1]))/2+pad

def num_static_fig(p, targets):
    HC,O4,As,Bs=p['HC'],p['O4'],p['As'],p['Bs']
    xc,yc,span=compute_range([targets,As,Bs,[HC],[O4]])
    fig=go.Figure(); th=np.linspace(0,2*np.pi,200)
    fig.add_trace(go.Scatter(x=HC[0]+p['r_crank']*np.cos(th),y=HC[1]+p['r_crank']*np.sin(th),
        mode='lines',line=dict(color='rgba(29,158,117,0.2)',width=1,dash='dot'),name='Crank circle'))
    fig.add_trace(go.Scatter(x=O4[0]+p['r_rocker']*np.cos(th),y=O4[1]+p['r_rocker']*np.sin(th),
        mode='lines',line=dict(color='rgba(55,138,221,0.2)',width=1,dash='dot'),name='Rocker circle'))
    for i in range(4):
        a=1.0 if i==0 else 0.3; sl=(i==0)
        Ai,Bi,Ci=As[i],Bs[i],targets[i]
        fig.add_trace(go.Scatter(x=[HC[0],Ai[0]],y=[HC[1],Ai[1]],mode='lines',
            line=dict(color=f'rgba(29,158,117,{a})',width=4 if sl else 2),showlegend=sl,name='Crank'))
        fig.add_trace(go.Scatter(x=[Ai[0],Ci[0],Bi[0],Ai[0]],y=[Ai[1],Ci[1],Bi[1],Ai[1]],
            mode='lines',fill='toself',fillcolor=f'rgba(127,119,221,{a*0.15})',
            line=dict(color=f'rgba(127,119,221,{a})',width=4 if sl else 2),showlegend=sl,name='Coupler'))
        fig.add_trace(go.Scatter(x=[Bi[0],O4[0]],y=[Bi[1],O4[1]],mode='lines',
            line=dict(color=f'rgba(55,138,221,{a})',width=4 if sl else 2),showlegend=sl,name='Rocker'))
    for i,(lbl,pt) in enumerate(zip(['C1','C2','C3','C4'],targets)):
        fig.add_trace(go.Scatter(x=[pt[0]],y=[pt[1]],mode='markers+text',text=[lbl],
            textposition='top center',showlegend=(i==0),name='Precision pts',
            marker=dict(size=14,color='#E24B4A',symbol='x',line=dict(width=3))))
    for lbl,pt,clr in [('HC',HC,'#1D9E75'),('O4',O4,'#378ADD')]:
        fig.add_trace(go.Scatter(x=[pt[0]],y=[pt[1]],mode='markers+text',text=[lbl],
            textposition='bottom center',marker=dict(size=16,color=clr,symbol='triangle-down'),name=lbl))
    fig.update_layout(**axis_layout(xc,yc,span))
    return fig, xc, yc, span

def graphical_fig(res):
    Cs,As,Bs=res['Cs'],res['As'],res['Bs']
    HR,HC,O4=res['HR'],res['HC'],res['O4']
    xc,yc,span=compute_range([Cs,As,Bs,[HR],[HC],[O4]]+[[Pi] for Pi in res['P']])
    fig=go.Figure(); th=np.linspace(0,2*np.pi,200)
    # Perp bisectors
    for p1,p2,col in [(Cs[0],Cs[2],'rgba(220,80,80,0.5)'),
                       (As[0],As[2],'rgba(80,150,220,0.5)')]:
        m,pv=perp_bisector(p1,p2); ext=span*0.8
        fig.add_trace(go.Scatter(x=[m[0]-ext*pv[0],m[0]+ext*pv[0]],
                                  y=[m[1]-ext*pv[1],m[1]+ext*pv[1]],
            mode='lines',line=dict(color=col,width=1.5,dash='dash'),showlegend=False))
    # Circle(HR,R)
    Rv=np.linalg.norm(As[0]-HR)
    fig.add_trace(go.Scatter(x=HR[0]+Rv*np.cos(th),y=HR[1]+Rv*np.sin(th),
        mode='lines',line=dict(color='rgba(200,120,0,0.4)',width=1,dash='dot'),name='Circle(HR,R)'))
    # Small r circles around C1, C3
    for Ci in [Cs[0],Cs[2]]:
        fig.add_trace(go.Scatter(x=Ci[0]+res['r_val']*np.cos(th),y=Ci[1]+res['r_val']*np.sin(th),
            mode='lines',line=dict(color='rgba(220,80,80,0.25)',width=0.8,dash='dot'),showlegend=False))
    # Crank circle
    fig.add_trace(go.Scatter(x=HC[0]+res['r_crank']*np.cos(th),y=HC[1]+res['r_crank']*np.sin(th),
        mode='lines',line=dict(color='rgba(29,158,117,0.3)',width=1,dash='dot'),name='Crank circle'))
    # Rocker circle
    fig.add_trace(go.Scatter(x=O4[0]+res['r_rocker']*np.cos(th),y=O4[1]+res['r_rocker']*np.sin(th),
        mode='lines',line=dict(color='rgba(55,138,221,0.3)',width=1,dash='dot'),name='Rocker circle'))
    # 4 mechanism positions
    for i in range(4):
        a=1.0 if i==0 else 0.3; sl=(i==0)
        Ai,Bi,Ci=As[i],Bs[i],Cs[i]
        fig.add_trace(go.Scatter(x=[HC[0],Ai[0]],y=[HC[1],Ai[1]],mode='lines',
            line=dict(color=f'rgba(29,158,117,{a})',width=3 if sl else 1.5),showlegend=sl,name='Crank'))
        fig.add_trace(go.Scatter(x=[Ai[0],Ci[0],Bi[0],Ai[0]],y=[Ai[1],Ci[1],Bi[1],Ai[1]],
            mode='lines',fill='toself',fillcolor=f'rgba(127,119,221,{a*0.12})',
            line=dict(color=f'rgba(127,119,221,{a})',width=3 if sl else 1.5),showlegend=sl,name='Coupler'))
        fig.add_trace(go.Scatter(x=[Bi[0],O4[0]],y=[Bi[1],O4[1]],mode='lines',
            line=dict(color=f'rgba(55,138,221,{a})',width=3 if sl else 1.5),showlegend=sl,name='Rocker'))
    # Image points
    for i,Pi in enumerate(res['P']):
        fig.add_trace(go.Scatter(x=[Pi[0]],y=[Pi[1]],mode='markers+text',text=[f'P{i+1}'],
            textposition='top right',marker=dict(size=7,color='#EF9F27'),
            showlegend=(i==0),name='Image pts (Pᵢ)'))
    # Ai labels
    for i,(Ai,lbl) in enumerate(zip(As,['A1','A2','A3','A4'])):
        fig.add_trace(go.Scatter(x=[Ai[0]],y=[Ai[1]],mode='markers+text',text=[lbl],
            textposition='bottom left',
            marker=dict(size=8,color='#1D9E75' if i in [0,2] else '#5DCAA5'),showlegend=False))
    # Precision points
    for i,(lbl,pt) in enumerate(zip(['C1','C2','C3','C4'],Cs)):
        fig.add_trace(go.Scatter(x=[pt[0]],y=[pt[1]],mode='markers+text',text=[lbl],
            textposition='top center',showlegend=(i==0),name='Precision pts',
            marker=dict(size=14,color='#E24B4A',symbol='x',line=dict(width=3))))
    # Ground pivots
    for lbl,pt,clr in [('HC',HC,'#1D9E75'),('O4',O4,'#378ADD')]:
        fig.add_trace(go.Scatter(x=[pt[0]],y=[pt[1]],mode='markers+text',text=[lbl],
            textposition='bottom center',marker=dict(size=16,color=clr,symbol='triangle-down'),name=lbl))
    fig.add_trace(go.Scatter(x=[HR[0]],y=[HR[1]],mode='markers+text',text=['HR'],
        textposition='top center',marker=dict(size=10,color='#EF9F27'),name='HR'))
    fig.update_layout(**axis_layout(xc,yc,span,h=600))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [('num_solutions',[]),('num_sel',0),('gfx_found',False)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# PRECISION POINT INPUT — shared widget
# ══════════════════════════════════════════════════════════════════════════════
def precision_inputs(prefix):
    defs=[[-6.0,0.0],[-2.0,5.0],[2.0,5.0],[6.0,0.0]]
    st.markdown("**Precision points**")
    cols=st.columns(4)
    result=[]
    for i,(lbl,d) in enumerate(zip(['C1','C2','C3','C4'],defs)):
        with cols[i]:
            st.markdown(f"**{lbl}**")
            x=st.number_input("x",value=float(d[0]),step=0.1,key=f'{prefix}_{lbl}_x')
            y=st.number_input("y",value=float(d[1]),step=0.1,key=f'{prefix}_{lbl}_y')
            result.append(np.array([x,y]))
    return result

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_num, tab_gfx = st.tabs(["🔢  Numerical synthesis  (exact)", "📐  Graphical construction  (educational)"])

# ─────────────────────────── NUMERICAL TAB ───────────────────────────────────
with tab_num:
    st.markdown("Finds an **exact** 4-bar mechanism whose coupler curve passes through "
                "all four precision points using numerical optimisation.")

    targets_n = precision_inputs("num")

    if st.button("🔍  Synthesize mechanism", type="primary", key='num_btn'):
        prog=st.progress(0.0,text="Searching…")
        sols=num_solve(targets_n,n_trials=400,seed=0,
                       progress_cb=lambda v: prog.progress(min(v,1.0),
                           text=f"Searching… {int(v*100)}%"))
        prog.empty()
        st.session_state.num_solutions=sols
        st.session_state.num_sel=0
        if not sols:
            st.error("No mechanism found for these points. Try different precision point positions.")
        else:
            st.success(f"Found {len(sols)} mechanism(s). ✓")

    solutions=st.session_state.num_solutions
    if solutions:
        if len(solutions)>1:
            st.session_state.num_sel=st.selectbox(
                "Select solution",range(len(solutions)),
                format_func=lambda i:(f"Solution {i+1}  —  "
                    f"r_crank={solutions[i]['r_crank']:.2f}  "
                    f"r_AB={solutions[i]['r_AB']:.2f}  "
                    f"r_rocker={solutions[i]['r_rocker']:.2f}  "
                    f"error={solutions[i]['max_err']:.4f}"),
                index=st.session_state.num_sel,key='sol_sel')

        p=solutions[st.session_state.num_sel]
        mc=st.columns(5)
        mc[0].metric("Crank |HC─A|",  f"{p['r_crank']:.3f}")
        mc[1].metric("Coupler |A─C|", f"{p['r']:.3f}")
        mc[2].metric("Coupler |A─B|", f"{p['r_AB']:.3f}")
        mc[3].metric("Rocker |B─O4|", f"{p['r_rocker']:.3f}")
        mc[4].metric("Ground |HC─O4|",f"{p['ground']:.3f}")

        fig_s,xc,yc,span=num_static_fig(p,targets_n)
        st.plotly_chart(fig_s,use_container_width=True)

        st.markdown("**Animation**")
        ac=st.columns([2,2,1])
        spd=ac[0].slider("Speed (ms/frame)",8,120,20,4,key='spd')
        nfr=ac[1].slider("Frames per revolution",90,720,360,90,key='nfr')
        if ac[2].button("▶ Play",key='play'):
            with st.spinner("Building animation…"):
                frames,tx,ty=animate_frames(p,targets_n,n_frames=nfr,speed_ms=spd)
            if not frames:
                st.error("Mechanism cannot complete full rotation. Try another solution.")
            else:
                fig_a=go.Figure(data=frames[0].data,frames=frames,
                    layout=go.Layout(**axis_layout(xc,yc,span,h=650),
                        updatemenus=[dict(type='buttons',showactive=False,
                            y=1.1,x=0.5,xanchor='center',
                            buttons=[
                                dict(label='▶ Play',method='animate',
                                     args=[None,dict(frame=dict(duration=spd,redraw=True),
                                                     fromcurrent=True,mode='immediate')]),
                                dict(label='⏸ Pause',method='animate',
                                     args=[[None],dict(frame=dict(duration=0,redraw=False),
                                                       mode='immediate')]),
                            ])],
                        sliders=[dict(steps=[dict(method='animate',
                            args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True))],
                            label='') for f in frames],
                            transition=dict(duration=0),x=0.05,y=0,len=0.9,
                            currentvalue=dict(visible=False))]))
                st.plotly_chart(fig_a,use_container_width=True)
                import pandas as pd
                tr=np.column_stack([tx,ty])
                rows=[{'Point':lbl,
                       'Min dist from trace':f"{np.linalg.norm(tr-T,axis=1).min():.5f}",
                       'Status':'✓' if np.linalg.norm(tr-T,axis=1).min()<0.1 else '✗'}
                      for lbl,T in zip(['C1','C2','C3','C4'],targets_n)]
                st.subheader("Verification")
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ─────────────────────────── GRAPHICAL TAB ───────────────────────────────────
with tab_gfx:
    st.info(
        "**Why the graphical method is approximate**  \n"
        "The Burmester method gives *exact* 3-position synthesis. "
        "When a 4th precision point is added, B₁ is determined from only "
        "P₁, P₂, P₃ (their circumcenter), so B₄ is not guaranteed to lie "
        "on the rocker circle — this gap is the **closure error** shown below.  \n"
        "The Numerical tab solves all 4 positions simultaneously and is always exact.",
        icon="📐")

    targets_g=precision_inputs("gfx")

    # ── Auto-find button ──────────────────────────────────────────────────
    st.markdown("**Step 1 — find valid starting parameters**")
    if st.button("⚙️  Auto-find valid parameters", key='gfx_auto'):
        with st.spinner("Searching (~5 sec)…"):
            best,err=auto_find_graphical(targets_g)
        hr_f,R_f,r_f,hc_f,br_f=best
        # Write directly to slider session_state keys so sliders update on rerun
        st.session_state['g_hr']=float(hr_f)
        st.session_state['g_R'] =float(R_f)
        st.session_state['g_r'] =float(r_f)
        st.session_state['g_hc']=float(hc_f)
        st.session_state['g_bA1']=int(br_f[0])
        st.session_state['g_bA3']=int(br_f[1])
        st.session_state['g_bA2']=int(br_f[2])
        st.session_state['g_bA4']=int(br_f[3])
        st.session_state.gfx_found=True
        st.success(f"Found parameters with closure error = {err:.4f}. "
                   "Sliders updated — adjust them freely to explore the construction.")
        st.rerun()

    # ── Sliders — always read from session_state keys ─────────────────────
    st.markdown("**Step 2 — adjust sliders to explore**")
    sc1,sc2=st.columns(2)
    with sc1:
        hr_t =st.slider("HR on perp-bisector of C1─C3",-20.0,20.0,
                         st.session_state.get('g_hr',3.0),0.05,key='g_hr')
        R_val=st.slider("R  (distance HR → Ai)",0.5,35.0,
                         st.session_state.get('g_R',8.0),0.05,key='g_R')
    with sc2:
        r_val=st.slider("r  (distance Ai → Ci)",0.1,20.0,
                         st.session_state.get('g_r',3.0),0.05,key='g_r')
        hc_t =st.slider("HC on perp-bisector of A1─A3",-20.0,20.0,
                         st.session_state.get('g_hc',2.0),0.05,key='g_hc')

    bc=st.columns(4)
    bA1=bc[0].selectbox("A1 branch",[0,1],
                         index=st.session_state.get('g_bA1',0),key='g_bA1',
                         help="Which intersection point to use for A1")
    bA3=bc[1].selectbox("A3 branch",[0,1],
                         index=st.session_state.get('g_bA3',0),key='g_bA3')
    bA2=bc[2].selectbox("A2 branch",[0,1],
                         index=st.session_state.get('g_bA2',0),key='g_bA2')
    bA4=bc[3].selectbox("A4 branch",[0,1],
                         index=st.session_state.get('g_bA4',0),key='g_bA4')

    # ── Construct & display ────────────────────────────────────────────────
    gres,gerr=graphical_construct(targets_g,hr_t,R_val,r_val,hc_t,bA1,bA3,bA2,bA4)

    if gerr:
        st.error(f"Construction failed: {gerr}")
        if not st.session_state.gfx_found:
            st.caption("👉 Click **Auto-find valid parameters** above to get a working starting point.")
    else:
        cl=gres['closure_err']
        mc=st.columns(5)
        mc[0].metric("Crank |HC─A|", f"{gres['r_crank']:.3f}")
        mc[1].metric("Coupler |A─C|",f"{gres['r_val']:.3f}")
        mc[2].metric("Coupler |A─B|",f"{gres['r_AB']:.3f}")
        mc[3].metric("Rocker |B─O4|",f"{gres['r_rocker']:.3f}")
        mc[4].metric("Closure error", f"{cl:.4f}",
                      delta="good ✓" if cl<0.5 else "high — C4 only approximate ⚠",
                      delta_color="normal" if cl<0.5 else "inverse")

        if cl>1.0:
            st.warning(f"Closure error = {cl:.3f}: "
                       "the coupler curve will not pass accurately through C4. "
                       "Run Auto-find or adjust sliders to reduce it.")

        with st.expander("Construction steps explained",expanded=False):
            for s in [
                "**Step 1** — HR is placed on the perp-bisector of C1─C3 (red dashed line). "
                "Any point on this line is equidistant from C1 and C3.",
                "**Step 2** — A1 and A3 are the intersections of circle(HR, R) with "
                "circles(C1, r) and (C3, r). They are at distance R from HR and distance r from Ci.",
                "**Step 3** — HC is placed on the perp-bisector of A1─A3 (blue dashed line), "
                "so |HC─A1| = |HC─A3| = r_crank. A1 and A3 lie on the crank circle.",
                "**Step 4** — A2 and A4 are found on the crank circle at distance r from C2 and C4.",
                "**Step 5** — Image points Pᵢ = A1 + R(φ₁ − φᵢ) · (HR − Aᵢ) map HR into "
                "the body frame of position 1. They represent where HR 'appears' "
                "when the body is un-rotated back to position 1.",
                "**Step 6** — B1 = circumcenter(P1, P2, P3). The rocker pivot "
                "O4 = circumcenter(B1, B2, B3).",
                f"**Closure check** — For exact 4-position synthesis, B4 must also lie "
                f"on the rocker circle. The error here is **{cl:.4f}** units. "
                "This is zero only by coincidence — the numerical method always achieves zero.",
            ]: st.markdown(s)

        st.plotly_chart(graphical_fig(gres), use_container_width=True)

# ─────────── GRAPHICAL ANIMATION ───────────
        st.markdown("**Approximate Motion Preview**")
        st.caption("⚠️ Motion is approximate — graphical method does not exactly satisfy all 4 points.")

        if st.button("▶ Preview motion (approximate)", key='gfx_anim'):
            if gres['closure_err'] > 1.0:
                st.warning("Closure error is high — animation may look inaccurate.")

            with st.spinner("Building animation…"):
                fig_anim = animate_graphical(gres, targets_g)

            if fig_anim:
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.error("Animation failed — mechanism cannot complete motion.")