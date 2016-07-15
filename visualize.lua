require "torch"
require "image"
require "math"
require "nn"

local training_dataset, testing_dataset, classes, classes_names = dofile('loadDataset.lua')
model = torch.load('network.dat')



function accumalateTensor(ten)
	local aten = ten[{{1},{},{},{}}]
	local separator = 4
	for i=2,ten:size(1) do
		aten = torch.cat(aten,torch.Tensor(1,1,ten:size(3),separator):zero())
		aten = torch.cat(aten,ten[{{i},{},{},{}}])
	end

	return aten:resize(1,aten:size(3),aten:size(4))
end


function normalizeImage(im)
	return 	(im-im:min())/(im:max()-im:min())
end

function visualizeConvNet(frameNo)

	local frame = training_dataset[frameNo][1]
	local Folder = 'FeatureMapsVisualized/'
	local filtersName = "filters.png"
	local input = "input_%03d.png"
	local featureMaps = "featureMaps_%03d.png"
	local inputName = string.format(input,frameNo)
	local featureMapsName = string.format(featureMaps,frameNo)

	-------------------------------------------------------------   save input image
	image2save = frame:resize(1,frame:size(2),frame:size(3))
	image2save = normalizeImage(image2save)
	image.save(Folder .. inputName,image2save)
	-------------------------------------------------------------   visualize & save filters
	image2save = accumalateTensor(model:get(1).weight)
	image2save = normalizeImage(image2save)
	image.save(Folder .. filtersName,image2save)
	-------------------------------------------------------------   visualize & save Feature maps	
	res = model:get(1):forward(frame)
	res = res:view(res:size(1), 1, res:size(2), res:size(3))

	image2save = accumalateTensor(res)
	image2save = normalizeImage(image2save)
	image.save(Folder .. featureMapsName,image2save)  
end

--[[
Folder = 'FeatureMapsVisualized/'
local filtersName = "filters.png"
local input = "input_%03d.png"
local featureMaps = "featureMaps_%03d.png"
--itorch.image(image.load(Folder.. filtersName ))
g = torch.Tensor(1,40,100):zero()
for frameNo=1,100 do
    local inputName = string.format(input,frameNo)
    local featureMapsName = string.format(featureMaps,frameNo)
    
    f = torch.cat(image.load(Folder.. inputName ),image.scale(image.load(Folder.. featureMapsName ),936,40))
    g = torch.cat(g,f)
    --itorch.image(f)
end
]]--