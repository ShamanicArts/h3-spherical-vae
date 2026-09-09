"""Attach the existing 360 comparison UI to downloaded service responses."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output.resolve()
    scenes=[]
    for receipt_path in sorted(out.glob('*-receipt.json')):
        r=json.loads(receipt_path.read_text());label=receipt_path.name.removesuffix('-receipt.json')
        variants=[]
        for field,title in [('unrepaired_video','Before pole repair'),('video','Selected pole repairs'),('canonical_video','Ordinary final prediction, same decode context')]:
            video=out/f'{label}-{field}.mp4'
            if not video.exists():continue
            poster=video.with_suffix('.png')
            import av
            with av.open(str(video)) as container:
                first=next(container.decode(video=0));first.to_image().save(poster)
            variants.append({'name':title,'status':'Downloaded','url':video.name,'review_url':video.name,'poster_url':poster.name,'snapshot_urls':[]})
        if len(variants)<2:continue
        cfg=r['input']
        scenes.append({'id':label,'name':label.replace('-',' '),'width':cfg['width'],'height':cfg['height'],'frames':cfg['frames'],'fps':24,'variants':variants,'guides':[],
            'note':f"Repair top: {cfg['repair_top']} · bottom: {cfg['repair_bottom']} · repair text: {cfg['top_prompt']!r} / {cfg['bottom_prompt']!r}. Complete ERP, silent. Receipt: {receipt_path.name}"})
    if not scenes:raise ValueError('No downloaded service comparisons found')
    page=(ROOT/'viewer/index.html').read_text()
    payload=json.dumps({'scenes':scenes,'default_view':'free'}).replace('<','\\u003c')
    page=re.sub(r'(<script id="experiment" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],page,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>Spherical H3 endpoint comparisons</h1>',page,count=1)
    page=re.sub(r'<p class="intro">.*?</p>','<p class="intro">Complete ERP before and after the selected pole repairs. Look up, down, across the seam and through time.</p>',page,count=1)
    page=re.sub(r'<section class="plan">.*?</section>','<section class="plan"><strong>Small-mask video repair</strong><p>Independent top/bottom instructions. No scene-prompt expansion or geometry preprocessing. Downloaded service results; visual quality remains for review.</p></section>',page,count=1,flags=re.S)
    for ident,text in [('evidenceBadge',f'{len(scenes)} service comparisons'),('executionBadge','Native ERP outputs'),('batchBadge','124 frames · 24 fps')]:
        page=re.sub(f'(id="{ident}">).*?(</span>)',lambda m:m[1]+text+m[2],page,count=1)
    page=page.replace('href="evidence.json"',f'href="{sorted(out.glob("*-receipt.json"))[0].name}"').replace('href="technique.html"','href="https://github.com/ShamanicArts/h3-spherical-vae/blob/main/docs/service.md"')
    shutil.copytree(ROOT/'viewer/vendor',out/'vendor',dirs_exist_ok=True)
    (out/'index.html').write_text(page)
    print(out/'index.html')


if __name__=='__main__':main()
