"""Build the PDF from curated, hash-checked report-card assets. No model calls."""
from pathlib import Path
import hashlib,json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'docs/report-card';D=json.loads((DATA/'evidence.json').read_text())
for a in D['images'].values():
    assert hashlib.sha256((DATA/a['file']).read_bytes()).hexdigest()==a['sha256']
PDF=ROOT/'docs/H3-final-prediction-report-card.pdf'
c=canvas.Canvas(str(PDF),pagesize=(1000,800),pageCompression=1,invariant=1)
c.setTitle('H3 seam report card - circular decoding and final shifted prediction');c.setAuthor('ShamanicArts')
INK='#182D2A';MUTED='#53645F';GREEN='#13745F';LINE='#D6DFDA'
def text(s,x,y,size=12,bold=False,color=INK):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.setFillColor(HexColor(color));c.drawString(x,y,str(s))
def para(s,x,top,w=928,size=12,color=INK):
    p=Paragraph(s,ParagraphStyle('body',fontName='Helvetica',fontSize=size,leading=size*1.35,textColor=HexColor(color)))
    _,h=p.wrap(w,800);p.drawOn(c,x,top-h);return top-h

def page(n,title,sub):
    c.setFillColor(HexColor('#FAFBF8'));c.rect(0,0,1000,800,fill=1,stroke=0)
    text('H3 / SEAM REPORT CARD',36,768,10,True,GREEN)
    text(title,36,729,26,True);para(sub,36,708,928,12,MUTED)
    c.setStrokeColor(HexColor(LINE));c.line(36,35,964,35)
    text('ShamanicArts | 8 September 2026 | Experimental results',36,20,9,color=MUTED)
    text(f'{n} / 4',941,20,9,color=MUTED)

def image(key,x,y,w,h):
    c.drawImage(str(DATA/D['images'][key]['file']),x,y,width=w,height=h)

def rule(y):c.setStrokeColor(HexColor(LINE));c.line(36,y,964,y)

page(1,'Circular decoding + a final shifted prediction','Ordinary sampling builds the scene. One additional prediction sees the wrap as an interior region. Circular decoding reconstructs the final pixels.')
for i,(arm,dec,label) in enumerate([('control','native','Ordinary sampling / native decode'),('control','circular128','Ordinary sampling / circular 128'),('final','circular128','Final shifted prediction / circular 128')]):
    x=36+i*312;text(label,x,658,11,True);image(f'temple50-{arm}-{dec}-f097-erp',x,512,304,133)
para('Full ERP context: same temple frame 97. The wrap is at the left/right edges. The next page centres it in the view.',36,499,928,10,MUTED)
text('Where it touches',36,456,16,True)
para('At the <b>final denoising step</b>, compute the ordinary prediction. Roll the in-progress video latent by half its width, predict again at the same noise level, then roll that prediction back. Accept only its first and last latent columns. Keep the ordinary prediction elsewhere and for audio.',36,437,928,12)
para('Complete the original final Euler update. Wrap opposite-edge context around the finished video latent, run the native VAE decoder, then crop the added margins. The shift covers the <b>whole clip</b>, not its last video frame.',36,369,928,12)
text('Tested recipe',36,313,13,True)
para('One latent column per edge (16 output pixels) | 128 px decoding context per edge | LoRA 1.0 | H3 BF16 | VAE FP16',36,295,928,11)
text('Incremental gain beyond circular decoding alone',36,256,14,True)
cols=[36,214,392,491,640,815]
for x,label in zip(cols,['Scene','Output / frames','Steps','Control MAE','Final-shift MAE','Reduction']):text(label,x,231,10,True)
for row,(key,name) in enumerate([('temple50','Temple'),('forest50','Forest'),('temple100','Temple')]):
    a=D['cases'][key];m=a['boundary_mae'];y=207-row*25
    values=[name,f'{a["width"]} x {a["height"]} / {a["frames"]}',a['steps'],f'{m["ordinary_circular128"]:.4f}',f'{m["final_shift_circular128"]:.4f}',f'{a["incremental_boundary_error_reduction_percent"]:.2f}%']
    for x,value in zip(cols,values):text(value,x,y,11)
    rule(y-8)
para('Boundary MAE: absolute RGB difference between the first and last columns, averaged over rows, channels and every frame (0-255). These percentages are pixel-error reductions, <b>not perceived-quality improvements</b>. Each row compares the same circular-128 decoder.',36,130,928,11)
para('Cost of the method: one extra model evaluation (50 -> 51 or 100 -> 101). No new training or weight changes. The extra prediction took about 27 s in the native temple H200 test; loading and decoding are separate.',36,77,928,10,MUTED)
c.showPage()

page(2,'Native-resolution temple: inspect the join','1536 x 672 | 243 frames | 50 steps | frame 97. Identical projections of verified PNGs. No sharpening or generated enhancement.')
for top,view,title in [(660,'eye75','Context / 75-degree view / eye level'),(449,'eye20','Close-up / 20-degree view / eye level'),(238,'up75','Context / 75-degree view / looking up 45 degrees')]:
    text(title,36,top,12,True)
    for i,(arm,dec,label) in enumerate([('control','native','Native decoder'),('control','circular128','Circular decoder 128'),('final','circular128','Circular 128 + final shifted prediction')]):
        x=36+i*312;text(label,x,top-20,10,True);image(f'temple50-{arm}-{dec}-f097-{view}',x,top-198,304,171)
c.showPage()

page(3,'100 steps: the final placement works best','1024 x 448 | 124 frames | same temple seed. Each variant starts from the same saved late trajectory; the circular-128 decoder stays fixed.')
cols=[36,235,390,600,810]
for x,label in zip(cols,['Correction step','Sigma','Ordinary steps afterward','Boundary MAE','Reduction']):text(label,x,660,11,True)
rows=[['No correction','-','-',f'{D["cases"]["temple100"]["boundary_mae"]["ordinary_circular128"]:.4f}','-']]
for a in sorted(D['timing'],key=lambda x:x['step']):rows.append([str(a['step'])+(' (final)' if a['step']==100 else ''),f'{a["sigma"]:.6f}',a['ordinary_steps_after'],f'{a["boundary_mae"]:.4f}',f'{a["reduction_percent"]:.2f}%'])
for i,row in enumerate(rows):
    y=637-i*24
    for x,v in zip(cols,row):text(v,x,y,11,bold=i==3)
    rule(y-8)
para('Step 99 uses the same noise level as the 50-step schedule\'s final step. Final placement still wins here. Sigma, update size and subsequent predictions change together; their individual contributions are not isolated.',36,540,928,11,MUTED)
for top,frame in [(493,0),(268,123)]:
    for x,arm,label in [(36,'control','Circular 128 / ordinary sampling'),(508,'final','Circular 128 / correction at step 100')]:
        text(f'{label} / frame {frame}',x,top,11,True)
        image(f'temple100-{arm}-circular128-f{frame:03d}-eye30',x,top-202,456,190)
para('30-degree views centred on the wrap. Residual structural mismatch remains; pixel matching alone does not establish an invisible seam.',36,53,928,9,MUTED)
c.showPage()

page(4,'Forest transfer and the current boundary of evidence','1536 x 672 | 243 frames | 50 steps | frame 97. A second scene: the additional boundary-error reduction is 9.10%, with some foliage softness.')
for x,arm,label in [(36,'control','Circular 128 / ordinary sampling'),(508,'final','Circular 128 / final shifted prediction')]:
    text(label,x,655,12,True);image(f'forest50-{arm}-circular128-f097-erp',x,443,456,199.5)
    text('Seam-centred 75-degree view',x,421,10,color=MUTED);image(f'forest50-{arm}-circular128-f097-eye75',x,155,456,256.5)
para('<b>Verified:</b> exact canonical final-step counterfactuals; unchanged exterior video latent and audio; repeated 100-step control matches every decoded frame. Layout is retained in inspected views. Residual seams and local texture changes remain.',36,133,450,10)
para('<b>Still open:</b> more independent scenes, 100 steps at native resolution, image conditioning, refinement/upscaling, long continuation and poles. This report does not establish those workflows.',508,133,456,10)
text('Method, settings, image hashes and limitations: github.com/ShamanicArts/h3-spherical-vae',36,56,10,True,GREEN)
c.linkURL('https://github.com/ShamanicArts/h3-spherical-vae/blob/main/docs/final-prediction.md',(36,51,900,69),relative=0)
c.showPage();c.save()
# Lossless PNG-style row prediction reduces PDF size without changing pixels.
import zlib
import numpy as np
from pypdf import PdfReader,PdfWriter
from pypdf.generic import NameObject,NumberObject,DictionaryObject
reader=PdfReader(PDF);writer=PdfWriter();writer.clone_document_from_reader(reader)
def signatures(document):
    result=[]
    for page in document.pages:
        for ref in page['/Resources'].get('/XObject',{}).values():
            obj=ref.get_object()
            if obj.get('/Subtype')=='/Image':result.append(hashlib.sha256(obj.get_data()).hexdigest())
    return sorted(result)
original=signatures(reader)
for page in writer.pages:
    for ref in page['/Resources'].get('/XObject',{}).values():
        obj=ref.get_object()
        if obj.get('/Subtype')!='/Image' or obj.get('/ColorSpace')!='/DeviceRGB' or obj.get('/BitsPerComponent')!=8:continue
        width,height=int(obj['/Width']),int(obj['/Height'])
        raw=np.frombuffer(obj.get_data(),dtype=np.uint8).reshape(height,width*3)
        filtered=raw.copy();filtered[:,3:]=raw[:,3:]-raw[:,:-3]
        rows=np.concatenate([np.ones((height,1),dtype=np.uint8),filtered],axis=1)
        obj._data=zlib.compress(rows.tobytes(),9)
        obj[NameObject('/Filter')]=NameObject('/FlateDecode')
        obj[NameObject('/DecodeParms')]=DictionaryObject({NameObject('/Predictor'):NumberObject(15),NameObject('/Colors'):NumberObject(3),NameObject('/BitsPerComponent'):NumberObject(8),NameObject('/Columns'):NumberObject(width)})
        obj.decoded_self=None
compact=PDF.with_suffix('.tmp.pdf')
with compact.open('wb') as handle:writer.write(handle)
assert signatures(PdfReader(compact))==original,'Lossless PDF image recompression changed pixels'
compact.replace(PDF);print(PDF,PDF.stat().st_size)
