"""Render an offline seven-minute pitch from the validated evidence packet."""
import html
import json
import os
from pathlib import Path
import sys
from io import StringIO

os.environ.setdefault('MPLCONFIGDIR', '/tmp/novo-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'novo-pulse-pitch'
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'backend'))
from apps.pol4.services import jury_evidence

p = jury_evidence()
percent = lambda value: f'{value*100:.2f}%'
esc = html.escape

def chart():
    rows = p['performance']['folds']
    fig, ax = plt.subplots(figsize=(11,3.2))
    xs = list(range(len(rows)))
    ax.bar([x-.18 for x in xs], [row['baseline']*100 for row in rows], .34, label='Pickup baseline', color='#94a3b8')
    ax.bar([x+.18 for x in xs], [row['champion']*100 for row in rows], .34, label='Submitted model', color=['#d97706' if row['shock_overlap'] else '#2563eb' for row in rows])
    ax.set_xticks(xs, [row['cutoff'] for row in rows]); ax.set_ylabel('WAPE (%)')
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); out=StringIO(); fig.savefig(out, format='svg', metadata={'Date': None}); plt.close(fig)
    return '\n'.join(line.rstrip() for line in out.getvalue()[out.getvalue().index('<svg'):].splitlines())

city = p['decisions'][0]
best_cluster = min(p['clustering']['levels'], key=lambda row: row['clustered_wape'])
slides = [
('0:00–0:45', 'The problem', 'See demand forming.<br><em>Decide with evidence.</em>',
 '<p class="lead">An earlier, auditable destination review from incomplete search interest.</p><div class="tags"><span>321 cities</span><span>7 provinces</span><span>30 check-in dates</span></div>',
 'The user is a destination growth analyst. We forecast search interest, not bookings or revenue. Describe the decision before describing the model.'),
('0:45–1:30', 'The formulation', 'Two clocks.<br>One observed-demand floor.',
 '<div class="formula">Final searches = observed + forecast remaining</div><p class="lead">Search date tells us what was knowable.<br>Check-in date tells us what we are forecasting.</p><p>Two LightGBM horizon bands · guarded remainder calibration · validated 9,630-row submission</p>',
 'Explain that already observed searches cannot be predicted away. Fold-cutoff safety is tested; historical city statistics in the submitted model retain a within-training-origin limitation, discussed in Q&A.'),
('1:30–2:45', 'Accuracy + disclosure', 'Measured improvement.<br>The difficult fold stays in.',
 f'<div class="metrics"><div><b>{percent(p["performance"]["champion"]["wape"])}</b><span>Five-fold WAPE</span></div><div><b>{percent(p["performance"]["baseline"]["wape"])}</b><span>Pickup baseline</span></div><div><b>{percent(p["shock"]["without_fold_wape"])}</b><span>Without shock fold · diagnostic only</span></div></div>'+chart()+f'<p class="disclosure">May 21 fold overlaps June 13–24, 2025 war; {percent(p["shock"]["error_share"])} of error. Temporal association, not causal proof. WAPE 95% fold CI: {percent(p["performance"]["confidence_interval"]["lower"])}–{percent(p["performance"]["confidence_interval"]["upper"])}.</p>',
 'State both scores together. The headline retains all five folds. The interval measures uncertainty of pooled WAPE, not a prediction interval for a city. Earlier-fold calibration selection gives WAPE 0.148618.'),
('2:45–3:40', 'Scientific judgment', 'Clustering had to earn its place.',
 f'<div class="three"><article><b>{best_cluster["unclustered_wape"]:.6f}</b><p>City model<br>Original three-fold sweep</p></article><article><b>{best_cluster["control_wape"]:.6f}</b><p>Same city predictions<br>Summed onto merged groups</p></article><article><b>{best_cluster["clustered_wape"]:.6f}</b><p>Cluster model<br>On those same groups</p></article></div><p class="lead">About 90% of the apparent gain was mechanical error cancellation.</p><p>We retain city-level submission rows. The 0.06% blend gain fails the greater-than-1% practical margin.</p><p class="disclosure">Three-fold uncalibrated experiment: compare within this slide only, not with the five-fold headline.</p>',
 'The control is the key contribution. Explain taking absolute errors after summing. Small gains and an exploratory sweep do not establish statistical superiority.'),
('3:40–5:10', 'Product demonstration', 'One destination.<br>A review with an owner.',
 f'<div class="review"><span>Example from the historical competition snapshot</span><h2>{esc(city["city"])}</h2><p>{city["predicted_demand"]:,.0f} forecast searches · {percent(city["observed_share"])} observed / forecast</p><p><b>Owner:</b> {esc(city["owner"])}</p><p><b>Action:</b> {esc(city["action"])}</p><p><b>Decision gate:</b> {esc(city["required_before_spend"])}</p></div><p><a href="http://localhost:3000/jury" target="_blank">Open the live evidence room →</a> · filter a province · select a city · export the queue</p>',
 'Spend no more than 90 seconds on the live demo. If the app is unavailable, the actual artifact-backed brief on this slide is the fallback. Never convert forecast searches into an invented revenue number.'),
('5:10–6:20', 'Business validation', 'Prove the workflow.<br>Then price the value.',
 '<div class="three"><article><b>Week 1</b><p>Define cohort, baseline analyst time, decision log and sponsor.</p></article><article><b>Week 2</b><p>Shadow current planning. Measure review time and corrections.</p></article><article><b>Weeks 3–4</b><p>Evaluate a preregistered pilot. Join bookings, inventory and costs before commercial claims.</p></article></div><p class="lead">Proposed gate: ≥20% lower median review time, with no rise in correction rate.</p><p>Buyer hypothesis: destination growth lead. Offer hypothesis: scoped evaluation, then a monthly workspace. Pricing and willingness-to-pay are unvalidated.</p>',
 'Explain that these are proposed pilot criteria, not achieved metrics. Economic worksheet uses measured labor hours and one consistent currency. Four weeks does not guarantee enough power for booking lift.'),
('6:20–7:00', 'Trust + next step', 'Inspectable evidence.<br>A concrete pilot ask.',
 '<ul><li>Champion trainset recovered: every original file hash matches.</li><li>Submitted forecast preserved; challengers evaluated separately.</li><li>Search-interest scope, shock sensitivity and model limitations disclosed.</li></ul><p class="lead">We seek an operator-sponsored pilot with timestamped booking, inventory and campaign-cost data.</p><p class="stop">7:00 — questions welcome.</p>',
 'Stop the pitch at seven minutes. Use the following Q&A appendix selectively; do not continue presenting it unprompted.'),
]
questions = [
 ('What has been proven commercially?', 'Forecast accuracy and a usable review workflow. No customer traction, booking lift or realized ROI is claimed. The four-week pilot measures planning time and review quality.'),
 ('Can you prove no leakage?', 'Horizon-indexed pickup features respect the forecast cutoff. The original city-history summaries include later within-fold training outcomes. The direct 66-feature origin-history challenger scores 0.151430 calibrated WAPE versus 0.147927 for the original setup on the same folds. Adversarial tests verify earlier features ignore later outcomes; pickup curves remain fold-fitted.'),
 ('Why keep the shock fold?', 'Dropping a fold because it was difficult is outcome-driven selection. Both all-fold and without-fold scores are disclosed; only the all-fold figure is the headline.'),
 ('Why not use the lowest challenger score?', 'The specifications and cluster height are exploratory, not nested. Promotion requires a prespecified rule and fresh closed holdout; a small retrospective gain is insufficient.'),
 ('What does the confidence interval mean?', 'The wide interval quantifies uncertainty of pooled WAPE over five folds. Experimental residual bands have measured forward-only coverage and no guarantee under regime shifts.'),
 ('Do you model lunar holidays?', 'No verified year-specific Iranian lunar-event table has been supplied or verified. We do not invent dates or encode unexpected war as a known-future holiday.'),
 ('What is the commercial ask?', 'A growth lead sponsor, an analyst, timestamped outcome/inventory/cost joins, and permission to run a preregistered evaluation. Willingness-to-pay and deployment costs must be measured.'),
]
body = ''
for i,(timing, label,title,content,note) in enumerate(slides):
    body += f'<section class="slide" data-slide="{i}"><header>NOVO PULSE / {esc(label)}<span>{timing} · {i+1}/7</span></header><h1>{title}</h1>{content}<aside class="notes">Speaker: {esc(note)}</aside></section>'
body += '<section class="slide" data-slide="7"><header>Q&A APPENDIX<span>7:00–14:00</span></header><h1>Answer first.<br>Then show the evidence.</h1><div class="questions">'+''.join(f'<details><summary>{esc(q)}</summary><p>{esc(a)}</p></details>' for q,a in questions)+'</div><p><a href="http://localhost:3000/jury">Open full experiment table and checksums →</a></p></section>'
css = '''*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#142b45;font:18px/1.5 system-ui,sans-serif}.slide{display:none;min-height:100vh;padding:42px 7vw 65px;max-width:1600px;margin:auto}.slide.active{display:block}header{display:flex;justify-content:space-between;letter-spacing:.12em;text-transform:uppercase;font-size:12px;color:#60758d}h1{font-size:clamp(32px,4.2vw,64px);line-height:1.08;letter-spacing:-.04em;margin:35px 0 28px}h1 em{font-style:normal;color:#2469bd}.lead{font-size:clamp(20px,2vw,29px);max-width:1050px}.tags,.metrics,.three{display:flex;gap:24px;margin:28px 0}.tags span{border:1px solid #bdcddd;padding:8px 18px;border-radius:25px;font-size:14px}.metrics>div,.three>article{flex:1;background:white;border-radius:14px;padding:20px}.metrics b,.three b{display:block;font-size:34px}.metrics span{font-size:14px;color:#536c87}svg{width:100%;height:230px;background:white;border-radius:12px}.disclosure{font-size:13px;border-left:3px solid #d97706;padding:10px 14px;background:#fff7e7}.formula{font-size:clamp(20px,2.6vw,38px);padding:30px;border-radius:14px;background:#16395c;color:white}.review{background:white;padding:28px;border-radius:14px;border-left:5px solid #2469bd;max-width:1000px}.review h2{font-size:34px;margin:10px 0}.review>span{font-size:12px;color:#60758d}.notes{display:none;padding:16px;background:#fef3c7;font-size:14px;margin-top:20px}body.notes-visible .notes{display:block}a{color:#2469bd}li{margin:16px 0}.stop{font-weight:bold;color:#2469bd}.questions{display:grid;grid-template-columns:1fr 1fr;gap:12px}.questions details{background:white;padding:16px;border-radius:10px;font-size:14px}.questions summary{font-weight:bold;cursor:pointer}nav{position:fixed;bottom:12px;right:20px;display:flex;align-items:center;gap:10px;font-size:12px}button{background:white;color:#142b45;border:1px solid #bdcddd;border-radius:8px;padding:8px 14px;cursor:pointer}button:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #2469bd;outline-offset:3px}@media(max-width:650px){.slide{padding:24px 18px 80px}.three,.metrics{gap:8px}.three>article,.metrics>div{padding:12px}.metrics b,.three b{font-size:22px}.questions{grid-template-columns:1fr}header{font-size:9px}.tags{flex-wrap:wrap;gap:8px}}@media print{.slide[data-slide="7"] h1{font-size:28px;margin:18px 0}.questions{gap:8px}.questions details{font-size:11px;padding:10px}.questions p{margin:8px 0;line-height:1.4}.slide{display:block;page-break-after:always;min-height:0;padding:18px}nav,.notes{display:none!important}h1{font-size:36px}details>p{display:block}.slide:last-of-type{page-break-after:auto}}'''
js = '''window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(item=>item.open=true));let index=0;const slides=[...document.querySelectorAll('.slide')];function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));document.getElementById('count').textContent=index<7?`${index+1}/7`:'Q&A';window.scrollTo(0,0)}document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);document.addEventListener('keydown',e=>{if(e.target.closest('button,summary,a'))return;if(['ArrowRight',' '].includes(e.key)){e.preventDefault();show(index+1)}if(e.key==='ArrowLeft')show(index-1);if(e.key.toLowerCase()==='n')document.body.classList.toggle('notes-visible');if(e.key.toLowerCase()==='p')window.print();if(e.key==='Home')show(0);if(e.key==='End')show(7)});show(0);'''
output = root / 'artifacts/pol4/pitch.html'
output.write_text('<!doctype html><html lang="en" dir="ltr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Novo Pulse · 7-minute pitch</title><style>'+css+'</style></head><body>'+body+'<nav><span>← → slides · N notes · P print</span><button id="prev" aria-label="Previous slide">←</button><span id="count"></span><button id="next" aria-label="Next slide">→</button></nav><script>'+js+'</script></body></html>')
print(output)
