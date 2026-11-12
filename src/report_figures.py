"""Regenerates report Figures 1-3 from the scored data. Run from the repo root: python src/report_figures.py"""
import pandas as pd, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
plt.rcParams.update({"font.family":["Inter","DejaVu Sans"],"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False})
OR,BL,GR="#E8692F","#4A90D9","#555555"
Path("outputs/figures").mkdir(parents=True,exist_ok=True)
d=pd.read_csv("data/processed/bank_churners_scored.csv")
base=d.churned.mean()

# Fig A: distributions
fig,ax=plt.subplots(1,3,figsize=(10,2.9),dpi=200)
spec=[("total_trans_ct","Transactions in last 12 months",None),("total_revolving_bal","Revolving balance ($)",None),("total_ct_chng_q4_q1","Transaction-count change, Q4 to Q1",(0,1.75))]
for a,(c,t,r) in zip(ax,spec):
    x=d[c] if r is None else d[c].clip(*r)
    bins=np.histogram_bin_edges(x,bins=30)
    a.hist(x[d.churned==0],bins=bins,density=True,color=BL,alpha=.65,label="Existing")
    a.hist(x[d.churned==1],bins=bins,density=True,color=OR,alpha=.65,label="Attrited")
    a.set_title(t,loc="left",fontsize=9.5,color=GR); a.set_yticks([]); a.grid(axis="x",color="#e5e5e5",lw=.6); a.set_axisbelow(True)
    m1,m0=d[c][d.churned==1].median(),d[c][d.churned==0].median()
    a.text(.98,.97,f"Median: attrited {m1:,.2f}\nexisting {m0:,.2f}".replace(".00","") if c!="total_ct_chng_q4_q1" else f"Median: attrited {m1:.2f}\nexisting {m0:.2f}",transform=a.transAxes,ha="right",va="top",fontsize=7.5,color=GR)
ax[0].legend(frameon=False,loc="center right",fontsize=8)
plt.tight_layout(); plt.savefig("outputs/figures/report_fig1_behaviour.png"); plt.close()

# Fig B: behavioural attrition rates
def bars(a,s,title,order=None):
    g=d.groupby(s,observed=True).churned.agg(["mean","size"])
    if order: g=g.loc[order]
    y=np.arange(len(g))[::-1]
    a.barh(y,g["mean"]*100,color=OR,height=.6)
    a.axvline(base*100,color="#888",ls="--",lw=1)
    a.set_yticks(y); a.set_yticklabels(g.index,fontsize=8.5)
    for yi,(m,n) in zip(y,g.values): a.text(m*100+1,yi,f"{m*100:.1f}%  (n={int(n):,})",va="center",fontsize=8,color=GR,bbox=dict(fc="white",ec="none",pad=1.2),zorder=5)
    a.set_xlim(0,60); a.set_title(title,loc="left",fontsize=9.5,color=GR); a.set_xticks([0,20,40,60]); a.set_xticklabels(["0%","20%","40%","60%"]); a.tick_params(left=False)
fig,ax=plt.subplots(2,2,figsize=(10,4.6),dpi=200)
bars(ax[0,0],"transaction_frequency_band","Transactions in 12 months",["Low (<=50)","Medium (51-80)","High (81+)"])
bars(ax[0,1],"utilisation_band","Card utilisation",["None (0%)","Low (0-30%)","Medium (30-60%)","High (60%+)"])
bars(ax[1,0],"products_held_band","Products held",["1-2 products","3-4 products","5-6 products"])
d["contacts_grp"]=d.contacts_count_12_mon.astype(int).astype(str)
d["contacts_grp"]=d.contacts_grp.map(lambda v:v+" contacts" if v!="1" else "1 contact")
bars(ax[1,1],"contacts_grp","Bank contacts in 12 months",[f"{k} contacts" if k!="1" else "1 contact" for k in ["0","1","2","3","4","5","6"]])
ax[1,1].set_xlim(0,130); ax[1,1].set_xticks([0,50,100]); ax[1,1].set_xticklabels(["0%","50%","100%"])
fig.text(.995,.005,"Dashed line: portfolio attrition 16.1%. Contacts panel uses a wider axis.",ha="right",fontsize=7.5,color=GR)
plt.tight_layout(rect=(0,.02,1,1)); plt.savefig("outputs/figures/report_fig2_behavioural_rates.png"); plt.close()

# Fig C: demographics, same axis
fig,ax=plt.subplots(1,3,figsize=(10,2.5),dpi=200)
age=["Under 35","35-44","45-54","55-64"]; d2=d[d.age_band.isin(age)]
def bars2(a,df,s,title,order):
    g=df.groupby(s,observed=True).churned.agg(["mean","size"]).loc[order]; y=np.arange(len(g))[::-1]
    a.barh(y,g["mean"]*100,color=OR,height=.6); a.axvline(base*100,color="#888",ls="--",lw=1)
    a.set_yticks(y); a.set_yticklabels(g.index,fontsize=8.5)
    for yi,(m,n) in zip(y,g.values): a.text(m*100+1,yi,f"{m*100:.1f}%  (n={int(n):,})",va="center",fontsize=8,color=GR,bbox=dict(fc="white",ec="none",pad=1.2),zorder=5)
    a.set_xlim(0,60); a.set_xticks([0,20,40,60]); a.set_xticklabels(["0%","20%","40%","60%"]); a.set_title(title,loc="left",fontsize=9.5,color=GR); a.tick_params(left=False)
bars2(ax[0],d2,"age_band","Age band",age)
bars2(ax[1],d,"income_segment","Income segment",["Low income","Lower-middle","Upper-middle/High","Unknown"])
bars2(ax[2],d,"gender","Gender",["F","M"])
plt.tight_layout(); plt.savefig("outputs/figures/report_fig3_demographics.png"); plt.close()
