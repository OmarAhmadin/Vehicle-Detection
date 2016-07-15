require "torch"
require "image"
require "math"
require "cunn"
require "cutorch" 

s = torch.load('RngState.dat')
torch.setRNGState(s)
--s= torch.getRNGState()
--torch.save('RngState.dat',s)
vehicle    = 1
nonVehicle = 2
classes 	  = { vehicle , nonVehicle } -- indices in torch/lua start at 1, not at zero
classes_names = {'vehicle','nonVehicle'}

w    = 100
h    = 40
trainToAllRatio = 0.9
ownFolder1 = 'OwnCollection/non-vehicles/MiddleClose/'
ownFolder2 = 'OwnCollection/vehicles/Right/'
ownFolder3 = 'OwnCollection/non-vehicles/Right/'
ownFolder4 = 'OwnCollection/non-vehicles/Left/'

ownFolder5 = 'OwnCollection/non-vehicles/Far/'

function shuffle(array)
    local counter = #array
    function swap(array, index1, index2)  array[index1], array[index2] = array[index2], array[index1] end

    while counter > 1 do
        local index = torch.random(counter)
        swap(array, index, counter)
        counter = counter - 1
    end
end

function catt(t1,t2)
	for i = 1,#t2 do
		t1[#t1+1] = t2[i]
	end
	return t1
end

function load_data_from_disk(folder)
	local dataset={}
	local  trainset, testset = {},{}
	--local labels = torch.Tensor(nNegAndPos,1):zero()

	local groupSize   = torch.Tensor{   500          ,    500       ,     500        ,	500	 ,    500         , 500		   }
	local groupClass  = torch.Tensor{ nonVehicle     ,  vehicle     ,   vehicle      , nonVehicle 	 ,   vehicle      , nonVehicle     }
	local groupFormat =             {'image%04d.png' ,'pos-%d.pgm'  , 'car_%04d.ppm' ,'image%04d.png','image%04d.png' ,'image%04d.png'}
	local groupFolder =             { ownFolder4     ,'TrainImages/', 'cars128x128/' ,  ownFolder1   , ownFolder2     ,  ownFolder3    }
	local grouptStart =             {      0         ,     0        ,      1         ,	 0 	 , 	0         ,	  0        }

	
	groupSize   = torch.cat(groupSize, torch.Tensor{    400        , 		400  });
	groupClass  = torch.cat(groupClass,torch.Tensor{    nonVehicle        ,    nonVehicle });
	groupFormat = 		     catt( groupFormat,{   'image%04d.png'       ,  'neg-%d.pgm' });
	groupFolder = 		     catt( groupFolder,{    ownFolder5        ,    'TrainImages/'} );
	grouptStart =                catt( grouptStart,{   0         ,  		0});
	


			   
	local datasetIdx  = 1;

	for groupIdx = 1,groupSize:size(1) do

	   for i = datasetIdx, datasetIdx + groupSize[groupIdx] -1 do
	      local filename = string.format( groupFormat[groupIdx] ,i-datasetIdx+grouptStart[groupIdx])
	      local input = image.load(groupFolder[groupIdx] .. filename)      -- images_set is global

	      if(input:size(1) == 3) then
	      		input = (input[{{1},{},{}}] + input[{{2},{},{}}] + input[{{3},{},{}}] ) / 3
	      end
	      if(groupIdx == 3) then
	      		input = input[{{},{33,96},{}}]
	      end

	      input = image.scale(input,w,h)	      

	      dataset[i] = {input:cuda(), groupClass[groupIdx]}-- class 2
	   end
	   
	   datasetIdx = datasetIdx + groupSize[groupIdx];

	end
	datasetLength = datasetIdx - 1

	shuffle(dataset)

	nTrain = trainToAllRatio * datasetLength
	nTest  = datasetLength   - nTrain
	
	for i = 1,nTrain do
		trainset[i] = dataset[i]
	end

	j=0;
	for i = 1,nTest do
		testset[i]  = dataset[nTrain+i]
		if testset[i][2] == 1 then		j=j+1 end
	end
	print(j)

	function trainset:size()  return nTrain end
	function testset:size()   return nTest end

	return trainset, testset 
end



local  trainset, testset = load_data_from_disk("TrainImages/")

return trainset, testset, classes, classes_names
