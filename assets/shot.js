const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  for (const [f, w, h, out] of [['results.html',1200,560,'results.png'],['thumb.html',1300,650,'thumbnail.png']]) {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: out==='thumbnail.png'?1:2 });
    await p.goto('file://' + process.cwd() + '/' + f); await p.waitForTimeout(300);
    await p.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
  }
  await b.close();
})();
