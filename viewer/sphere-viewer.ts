/** Compare adapter around the user's video-xr-player mono360 sphere factory.
 * The existing synchronized media transport supplies frames; no independent clocks.
 */
import * as THREE from 'three';

export function createRenderer(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false, preserveDrawingBuffer:true});
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1.6, .01, 1000);
  // Same mono sphere geometry and UV mapping as the original viewer.
  const geometry = new THREE.SphereGeometry(10,128,64,0,Math.PI*2,0,Math.PI);
  geometry.scale(-1,1,1);
  const material = new THREE.MeshBasicMaterial({toneMapped:false});
  scene.add(new THREE.Mesh(geometry, material));
  const textures = new Map<HTMLVideoElement, THREE.VideoTexture>();
  return {
    draw(video: HTMLVideoElement, view: {yaw:number,pitch:number,fov:number}, width=800, height=500) {
      let texture = textures.get(video);
      if (!texture) {
        texture = new THREE.VideoTexture(video);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.generateMipmaps = false;
        texture.wrapS = THREE.RepeatWrapping;
        textures.set(video, texture);
      }
      if (material.map !== texture) { material.map = texture; material.needsUpdate = true; }
      texture.needsUpdate = true;
      if (canvas.width !== width || canvas.height !== height) renderer.setSize(width,height,false);
      const aspect = width/height;
      // Shared FOV means horizontal FOV, matching the comparison's 75° views.
      camera.aspect = aspect;
      camera.fov = THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(view.fov)/2)/aspect));
      camera.updateProjectionMatrix();
      // Factory u=0/1 is +X. Yaw0 is the ERP join; yaw180 its opposite.
      const phi = -THREE.MathUtils.degToRad(view.yaw);
      const pitch = THREE.MathUtils.degToRad(Math.max(-89.99,Math.min(89.99,view.pitch)));
      camera.lookAt(Math.cos(phi)*Math.cos(pitch),Math.sin(pitch),Math.sin(phi)*Math.cos(pitch));
      renderer.render(scene,camera);
      canvas.dataset.camera = JSON.stringify(view);
    },
    release(video: HTMLVideoElement) { const texture=textures.get(video);if(texture){texture.dispose();textures.delete(video);if(material.map===texture)material.map=null;} },
    dispose() { for (const texture of textures.values()) texture.dispose(); textures.clear(); material.dispose(); geometry.dispose(); renderer.dispose(); renderer.forceContextLoss(); },
  };
}
