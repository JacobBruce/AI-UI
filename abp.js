class AudioBufferProcessor extends AudioWorkletProcessor
{ 
	constructor(options) {
		super(options);
		this.recBuffer = [];
		this.padSamples = [];
		this.avgCount = 0;
		this.avgSample = 0.0;
		this.minSample = 0.01;
		this.animTime = 50;
		this.maxIdleTime = 5000;
		this.maxMuteTime = 3000;
		this.maxRecTime = 300000;
		this.startTime = Date.now();
		this.lastTime = this.startTime;
		this.postTime = this.startTime;
		this.started = false;
		this.stopped = false;
		
		this.port.onmessage = (e) => {
			if (e.data.msg == "stop") {
				this.stop();
			}
		};
	}
	
	process(inputs, outputs) {
		if (this.stopped) return false;
		
		const input = inputs[0];
		const output = outputs[0];
		const nowTime = Date.now();

		input[0].forEach((sample, i) => {
			
			if (Math.abs(sample) > this.minSample) {
				
				if (!this.started) {
					if (this.padSamples.length > 999)
						this.padSamples = this.padSamples.slice(-999);
					this.recBuffer = this.padSamples;
					this.started = true;
					this.startTime = nowTime;
					this.port.postMessage({msg:"start"});
				}
				this.lastTime = nowTime;
			}
			
			if (this.started) {
				this.avgSample += Math.abs(sample);
				this.avgCount++;
				this.recBuffer.push(sample);
			}
		});
		
		if (this.started) {
			
			if (nowTime-this.lastTime > this.maxMuteTime || nowTime-this.startTime > this.maxRecTime) {
				this.stop();
				return false;
			}
			
			if (nowTime-this.postTime > this.animTime) {
				if (this.avgCount > 0) {
					this.port.postMessage({msg:"anim", sample:this.avgSample/this.avgCount});
					this.postTime = nowTime;
					this.avgCount = 0;
					this.avgSample = 0.0;
				}
			}
			
		} else {
			if (nowTime-this.startTime > this.maxIdleTime) {
				this.stop();
				return false;
			}
		}
		
		return true; 
	}
	
	stop() {
		this.started = false;
		this.stopped = true;
		this.port.postMessage({msg:"stop", buffer:this.recBuffer});
	}
}

registerProcessor('worklet-processor', AudioBufferProcessor);