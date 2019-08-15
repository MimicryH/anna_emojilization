function label = experi(wavPath)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%tic
%{
opts.gpus = 1 ;
opts.modelName = 'emovoxceleb-student';

buckets.pool = [2 5 8 11 14 17 20 23 27 30];
buckets.width  = [100 200 300 400 500 600 700 800 900 1000];
NUM=[52,22,9,11,30,45,4,19,12,32,41,22];

addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/mex/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/xtest/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/src/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/simplenn/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/+dagnn')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/matlab/+dagnn/@DagNN/')
addpath('E:\DoobiePJ\doobieBot/MatConvNet/matconvnet-1.0-beta25/contrib/mcnExtraLayers/matlab/')

addpath(genpath('E:\DoobiePJ\doobieBot\MatConvNet\matconvnet-1.0-beta25'))
%django

%mex -setup cpp
%vl_compilenn('enableGpu',true,'cudaMethod', 'nvcc', 'enableCudnn', true)
%vl_compilenn('enableGpu',true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0','cudaMethod', 'nvcc', 'enableCudnn', 'true','cudnnRoot', 'D:\DoobiePJ\NAVI\MatConvNet\matconvnet-1.0-beta25\local\cuda')
%vl_compilenn('EnableGpu',true)
run vl_setupnn
%which -all vl_simplenn
%which -all vl_nnconv
%vl_testnn('gpu',true)

if ~strcmp(opts.modelName, 'random')
		dag = emoVoxZoo(opts.modelName) ;
end

train_thresholds=[0.8639,0.07,0.0089,0.0246,0.0010]
unheard_thresholds=[0.8979,0.0556,0.0096,0.0212,9.1051e-04]
heard_thresholds=[0.8488,0.0779,0.0135,0.0403,9.7260e-04]


lossLayers = arrayfun(@(x) isa(x.block, 'dagnn.Loss'), dag.layers) ;
  removables = {dag.layers(lossLayers).name} ;
  for ii = 1:numel(removables)
    dag.removeLayer(removables{ii}) ;
  end
	dag.mode = 'test' ;
	if numel(opts.gpus), gpuDevice(opts.gpus) ; dag.move('gpu') ; end

	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;
    %save dag

%}
%toc

tic
%wav='/home/affcgroup/mustudy/data/lianke/13/';
%	fprintf('processing images with %s\n', opts.modelName) ;
	%for ii = 1:numIms
    %dags=load('dag.mat')
    %dag=dags.dag
    global struct buckets;
    global dag;
    global train_thresholds
    global unheard_thresholds
    global heard_thresholds
    label=zeros(1,5)
%    for ii = 1:22
		
        %aud=[num2str(ii),'.wav']
		%wavPath = imdb.tracks.wavPaths{ii} ;
        %wavPath=fullfile(wav,aud)

		inp1 = test_getinput(wavPath, buckets);
		s1 = size(inp1, 2);
		p1 = buckets.pool(s1 == buckets.width);
		ind1 = dag.getLayerIndex('pool6')

    if size(inp1, 2) > 0
      dag.layers(ind1).block.poolSize=[1 p1] ;
      %dag.eval({'data', gpuArray(inp1)}) ;
      dag.eval({'data', gpuArray(inp1)}) ;
      out = gather(squeeze(dag.vars(end).value)) % risky use of end variable

      normedLogits = vl_nnsoftmaxt(out', 'dim', 2)
      result=normedLogits(:,1:5)>=unheard_thresholds
      label(1,:)=result;
      %logits(ii, :) = out' ;
      rate = 1 / toc ;
      %etaStr = zs_eta(rate, ii, numIms) ;
      %fprintf('processed image %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
      %			 ii, numIms, rate, 100 * ii/numIms, etaStr) ;
     else
       fprintf('empty audio clip\n') ; label = zeros(1,5);
       %keyboard
    end

 %   end
    label
    toc
    %savepath='/home/affcgroup/mcnCrossModalEmotions-master'
    %save savepath label

end

