function inp = test_getinput(image, buckets)
% --------------------------------------------------
  audio.window   = [0 1];
  audio.fs       = 16000;
  audio.Tw       = 25;
  audio.Ts       = 10;            % analysis frame shift (ms)
  audio.alpha    = 0.97;          % preemphasis coefficient
  audio.R        = [300 3700];  % frequency range to consider
  audio.M        = 40;            % number of filterbank channels
  audio.C        = 13;            % number of cepstral coefficients
  audio.L        = 22;            % cepstral sine lifter parameter

	%audfile = [image(1:end-3),'wav'] ;
	%z	= audioread(audfile) ;
    
    z=audioread(image) ;
  addpath('/usr/local/MATLAB/R2017b/toolbox/MatConvNet/matconvnet-1.0-beta25/contrib/VGGVox/mfcc/')
  assert(size(z,2) <= 2, 'unexpected number of streams') ;
  z = z(:,1) ; % take left stream if stereo is available
	SPEC = runSpec(z, audio) ;
	mu = mean(SPEC, 2) ;
  stdev	= std(SPEC, [], 2) ;
  nSPEC	= bsxfun(@minus, SPEC, mu) ;
  nSPEC	= bsxfun(@rdivide, nSPEC, stdev) ;
  rsize	= buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last')) 
  rstart = round((size(nSPEC, 2) - rsize) / 2) ;
  if rstart == 0, rstart = 1 ; end
	inp(:,:) = gpuArray(single(nSPEC(:,rstart:rstart+rsize-1))) ;
end

