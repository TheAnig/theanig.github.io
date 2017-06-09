/*

Customized version of the bokeh effect found at : https://codepen.io/jackrugile/pen/gaFub

Author: TheAnig

*/

/* Variables - description

	c1, c2: canvas elements
	ctx1,2 : canvas contexts

	twopi: literally 2 * pi

	sizeBase: used for number of circles generated

	cw: canvas width
	ch: canvas height

	opt: options (explained below)

	count: number of circles
*/
var c1 = document.getElementById('c1'),
	ctx1 = c1.getContext( '2d' ),
	c2 = document.getElementById('c2'),
	ctx2 = c2.getContext( '2d' ),
	twopi = Math.PI * 2,
	sizeBase,
	cw,
	opt,
	hue,
	count;

function rand(min, max) {
	return Math.random() * (max - min) + min;
}

function hsla(h,s,l,a) {
	return 'hsla( ' + h + ', ' + s + '%, '+ l + '%, ' + a + ')';
}

function create() {


	sizeBase = cw + ch;


	count = Math.floor(sizeBase * 0.3),


	hue = rand(0, 360),

/*
	Options
	All the options act as an inclusive interval from which a value is chosen randomly

	radius: radii of circles
	blur: blur intensity
	hue, saturation, lightness(luminosity), alpha(transparency): self explanatory


*/

	opt = {
		radiusMin: 1,
		radiusMax: sizeBase * 0.03,
		blurMin: 5,
		blurMax: sizeBase * 0.04,
		hueMin: hue,
		hueMax: hue + 300,
		saturationMin: 50,
		saturationMax: 70,
		lightnessMin: 20,
		lightnessMax: 50,
		alphaMin: 0.1,
		alphaMax: 0.5
	}


	ctx1.clearRect( 0, 0, cw, ch );
	ctx1.globalCompositeOperation = 'lighter';

	//Generate random circles

	while(count--){
		var radius = rand( opt.radiusMin, opt.radiusMax ),
			blur = rand( opt.blurMin, opt.blurMax ),
			x = rand( 0, cw ),
			y = rand( 0, ch ),
			hue = rand( opt.hueMin, opt.hueMax ),
			saturation = rand( opt.saturationMin, opt.saturationMax ),
			lightness = rand( opt.lightnessMin, opt.lightnessMax ),
			alpha = rand( opt.alphaMin, opt.alphaMax );

		ctx1.shadowColor = hsla( hue, saturation, lightness, alpha );
		ctx1.shadowBlur = blur;
		ctx1.beginPath();
		ctx1.arc( x, y, radius, 0, twopi);
		ctx1.closePath();
		ctx1.fill();
	}
}

function loop(){
	requestAnimationFrame( loop );

	ctx2.clearRect( 0, 0, cw, ch );
	ctx2.globalCompositeOperation = 'source-over';
	ctx2.shadowBlur = 0;
	ctx2.drawImage(c1, 0, 0);
	ctx2.globalCompositeOperation = 'lighter';

	//Reserved for future post processing
}

function resize(){
	//Resize operation, creating a new bokeh everytime is wasteful, but looks better overall than resizing current one.
	cw = c1.width = c2.width = window.innerWidth,
	ch = c1.height = c2.height = window.innerHeight;
	create();
}

function click(){
	create();
}

function init(){
	resize();
	create();
	loop();
}

window.addEventListener( 'resize', resize );
window.addEventListener( 'click', click );

init();