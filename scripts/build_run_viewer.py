"""Attach the repository's synchronized 360 viewer to downloaded runtime outputs."""
import argparse,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve();r=json.loads((out/'result.json').read_text());cfg=r['config'];variants=[]
    # Put the main matched pair first; the existing UI activates only two videos.
    for context in (128,0,384):
        for arm in ('control','treatment'):
            if not any(d['arm']==arm and d['context']==context for d in r['decodes']):continue
            stem=f'{arm}-context{context}';video=out/(stem+'.mp4')
            if not video.is_file():raise ValueError(f'Missing downloaded video: {video.name}')
            variants.append({'name':f'{"Ordinary prediction" if arm=="control" else "Final shifted prediction"} / {"native decode" if context==0 else "circular "+str(context)}','status':'Verified','url':video.name,'review_url':video.name,'poster_url':stem+'-frame000.png','snapshot_urls':[p.name for p in sorted(out.glob(stem+'-frame*.png'))]})
    scene={'id':'clean-wheel','name':'Clean wheel / temple','width':cfg['width'],'height':cfg['height'],'frames':cfg['frames'],'fps':cfg.get('fps',24),'variants':variants,'guides':[],'note':f'{cfg["steps"]} steps + one final evaluation · BF16 H3/text · LoRA 1.0 · matched canonical forward · complete ERP, silent video.'}
    page=(ROOT/'viewer/index.html').read_text()
    payload=json.dumps({'scenes':[scene],'default_view':'free'}).replace('<','\\u003c')
    page=re.sub(r'(<script id="experiment" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],page,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>Clean deployment: matched seam comparison.</h1>',page,count=1)
    page=re.sub(r'<p class="intro">.*?</p>','<p class="intro">Fresh prompt-to-video generation from the repository wheel. Compare the final shifted prediction against its ordinary counterfactual. Inspect the seam and the opposite side.</p>',page,count=1)
    page=re.sub(r'<section class="plan">.*?</section>','<section class="plan"><strong>Final local prediction → circular decode</strong><p>One extra prediction, one accepted latent column at each edge. Both outputs share the same ordinary final forward call. Other latent columns and audio are checked for exact equality.</p></section>',page,count=1)
    for ident,value in [('evidenceBadge','1 matched deployment'),('executionBadge',f'{cfg["steps"]} steps · {cfg["width"]} × {cfg["height"]}'),('batchBadge',f'{cfg["frames"]} frames · 24 fps')]:page=re.sub(f'(id="{ident}">).*?(</span>)',lambda m:m[1]+value+m[2],page,count=1)
    page=page.replace('href="evidence.json"','href="result.json"').replace('href="technique.html"','href="https://github.com/ShamanicArts/h3-spherical-vae/blob/main/docs/final-prediction.md"')
    shutil.copytree(ROOT/'viewer/vendor',out/'vendor',dirs_exist_ok=True)
    (out/'index.html').write_text(page);print(out/'index.html')
if __name__=='__main__':main()
