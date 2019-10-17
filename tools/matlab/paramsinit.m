function [buckets,dag] = paramsinit()

tic
opts.gpus = 1 ;
opts.modelName = 'emovoxceleb-student';

global struct buckets;
global dag;
global train_thresholds
global unheard_thresholds
global heard_thresholds


buckets.pool = [2 5 8 11 14 17 20 23 27 30];
buckets.width  = [100 200 300 400 500 600 700 800 900 1000];
NUM=[52,22,9,11,30,45,4,19,12,32,41,22];
%{
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/mex/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/xtest/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/src/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/simplenn/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/+dagnn')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/matlab/+dagnn/@DagNN/')
addpath('D:\DoobiePJ\navi/MatConvNet/matconvnet-1.0-beta25/contrib/mcnExtraLayers/matlab/')
%}
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
		 dag = emoVoxZoo(opts.modelName);
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
    toc
end

