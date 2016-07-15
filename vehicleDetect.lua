require "torch"
require "nn"
require "math"
require "cunn"
require "cutorch"
require "image"

-- global variables
DataMean = 0
DataStd  = 0;
w = 100
h = 40
vehicle    = 1
nonVehicle = 2
torch.manualSeed(1)                  
network = torch.load('network.dat')
network=network:cuda()
function predict(network,input,confidence)
         local inp    = torch.CudaTensor(1,h,w):zero()
         if input:size(3) ~= 100 then
            return 1
         end
         if input:size(2) ~= 40 then
            return 1
         end

         inp[1]       = input

         --inp[{ {}, {}, {}  }]:add(-DataMean)    -- mean subtraction
         --inp[{ {}, {}, {}  }]:div(DataStd)      -- std scaling

         local responses_per_class     =  network:forward(inp) 
         local probabilites_per_class  = torch.exp(responses_per_class)
         local probability, prediction = torch.max(probabilites_per_class, 1)
   
         --output =  prediction[1];
         output = 1;
         if probabilites_per_class[1]<confidence then
          output =2;
         end
         
         return output,probabilites_per_class[1]
end



function loadAndTestFrame(frameNo,confidence,mode)

        local name_format= "test_%03d.png"
	--local name_format2 = "test_%03d.png"
        local folder = 'FinalExp/inputFrames/'
        local folder1 = 'FinalExp/outputFrames/'
        local folder5 = 'FinalExp/BS_outputFrames/'


        local filename = string.format(name_format,frameNo)
	
--        local filename2 = string.format(name_format2,frameNo)

        frame = image.load(folder .. filename)      -- images_set is global

	    if(frame:size(1) == 3) then
 		     frame = (frame[{{1},{},{}}] + frame[{{2},{},{}}] + frame[{{3},{},{}}] ) / 3
        end

        frameProcessed,frameProcessedUsingBS = processFrame(network,frame,confidence,frameNo,mode);
        image.save(folder1 .. filename,frameProcessed)
        image.save(folder5 .. filename,frameProcessedUsingBS)
end

function processFrame(network,frame,confidence,frameNo,mode)
         --frame[{ {}, {}, {}  }]:add(-DataMean)    -- mean subtraction
         --frame[{ {}, {}, {}  }]:div(DataStd)      -- std scaling
     local folder2 = 'FinalExp/BS_inputFrames/'
     local folder3 = 'FinalExp/BS_outputFrames/'
     local folder4 = 'FinalExp/BS_processedFrames/'
     local name_format = "test_%03d.png"
     local filename = string.format(name_format,frameNo)


     local ref    = torch.Tensor(1000,4)
     local refc   = 1	
     local framet = torch.Tensor(frame)
	framet:cuda()       
     local window = {}
     local y = 0
     local th = 2
     ww=200
     wh =100
--[[local frameProcessed = torch.CudaTensor(3,frame:size(2),frame:size(3)):zero()
     frameProcessed[{{1},{},{}}] = frame
     frameProcessed[{{2},{},{}}] = frame
     frameProcessed[{{3},{},{}}] = frame
    
     n = frame:size(2)
     m = frame:size(3)
     
 local frameProcessedUsingBS = frameProcessed:clone()
    
for wi=1,#ww do --]]
     ------------------------------------------------------- Background subtraction
     if mode ~= 'normal' then
         BS = image.load(folder2 .. filename,1)
         BS1 = -(BS-1)
         mask= torch.Tensor(wh,ww):zero()+1/(ww*wh)  --
         BS2 = image.convolve(BS1, mask,'same');
         BS3 = BS2:ge(0.9)   
         --ratio = BS3:sum() / BS3:nElement()
         --print('ratio = ',1-ratio)

         BS3[{{},{1,torch.round(ww/2)}}] = 1
         BS3[{{},{-torch.round(ww/2),-1}}] = 1
         BS3[{{1,torch.round(wh/2)},{}}] = 1
         BS3[{{-torch.round(wh/2),-1},{}}] = 1
         BS4 = torch.Tensor(1,BS3:size(1),BS3:size(2))
         BS4[{{1},{},{}}] = BS3

         image.save(folder4 .. filename,BS4)
     else
        BS3 = torch.Tensor(frame:size(2),frame:size(3)):zero()
     end
     -------------------------------------------------------
--
     local frameProcessed = torch.CudaTensor(3,frame:size(2),frame:size(3)):zero()
     frameProcessed[{{1},{},{}}] = frame
     frameProcessed[{{2},{},{}}] = frame
     frameProcessed[{{3},{},{}}] = frame
    
     n = frame:size(2)
     m = frame:size(3)
     --]]
     s = 5;
     
     for i=1,n-wh,s do
        for j=1,m-ww,s do
    	        resizedFrame = image.scale(framet[{{1}, {i,i+wh-1},{j,j+ww-1} }],100,40,'simple')
                y,prob = predict(network,  resizedFrame ,confidence) 
                if y == 1 then
                    if ((mode == 'BS') and (BS3[i+wh/2][j+ww/2] ==0)) or (mode ~= 'BS') then
                       
        	            flag = 1;

                  	    for k=1,(refc-1) do
                	        d = ((ref[k][1] - i)^2 + (ref[k][2] - j)^2)^0.5
                    		if d <(wh/1.2) then      --wh or wh/2 ...
                               if prob<=ref[k][3] then
                    		        flag = 0                       -- set to 1 if u want to cancel averaging
                               else
                                    ref[k][3] = 0 
                               end
                    		end
            	        end

            		    if(flag == 1) then
            			    ref[refc][1] = i
            			    ref[refc][2] = j
            			    ref[refc][3] = prob
                            ref[refc][4] = BS3[i+wh/2][j+ww/2]  --(0) ==> using Background subtraction
            			    refc = refc+1
            		    end
                    end
                end
        end
    end
    
  local frameProcessedUsingBS = frameProcessed:clone()
    ----------------------------------------------------------------
    nWindows   = 0
    nWindowsBS = 0
    for k = 1,(refc-1) do
        i = ref[k][1] 
        j = ref[k][2]
        if (ref[k][3] > 0) and (ref[k][4] == 0)  then

            frameProcessedUsingBS[{{}, {i,i+th},{j,j+ww-1} }]      = 0;
            frameProcessedUsingBS[{{}, {i,i+wh-1},{j,j+th} }]      = 0;
            frameProcessedUsingBS[{{}, {i+wh-1,i+wh-1+th},{j,j+ww-1} }]  = 0;
            frameProcessedUsingBS[{{}, {i,i+wh-1},{j+ww-1,j+ww-1+th} }]  = 0;

            frameProcessedUsingBS[{{2}, {i,i+th},{j,j+ww-1} }]      = 1;
            frameProcessedUsingBS[{{2}, {i,i+wh-1},{j,j+th} }]      = 1;
            frameProcessedUsingBS[{{2}, {i+wh-1,i+wh-1+th},{j,j+ww-1} }]  = 1;
            frameProcessedUsingBS[{{2}, {i,i+wh-1},{j+ww-1,j+ww-1+th} }]  = 1;
            nWindowsBS = nWindowsBS+1
               --frameProcessedUsingBS[{{2},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 1
               --frameProcessedUsingBS[{{1},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 0
               --frameProcessedUsingBS[{{3},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 0
       end
    end
    

    for k = 1,(refc-1) do
        i = ref[k][1] 
        j = ref[k][2]
        if ref[k][3] > 0 then
            frameProcessed[{{}, {i,i+th},{j,j+ww-1} }]      = 0;
            frameProcessed[{{}, {i,i+wh-1},{j,j+th} }]      = 0;
            frameProcessed[{{}, {i+wh-1,i+wh-1+th},{j,j+ww-1} }]  = 0;
            frameProcessed[{{}, {i,i+wh-1},{j+ww-1,j+ww-1+th} }]  = 0;

            frameProcessed[{{2}, {i,i+th},{j,j+ww-1} }]      = 1;
            frameProcessed[{{2}, {i,i+wh-1},{j,j+th} }]      = 1;
            frameProcessed[{{2}, {i+wh-1,i+wh-1+th},{j,j+ww-1} }]  = 1;
            frameProcessed[{{2}, {i,i+wh-1},{j+ww-1,j+ww-1+th} }]  = 1;
            nWindows = nWindows+1
               --frameProcessed[{{2},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 1
               --frameProcessed[{{1},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 0
               --frameProcessed[{{3},{i+wh/2,i+wh/2+5},{j+ww/2,j+ww/2+5}}] = 0
       end
    end
--end
    print('nWindows    =',  nWindows)
    print('nWindowsBS  =', nWindowsBS)

    return frameProcessed,frameProcessedUsingBS
end
