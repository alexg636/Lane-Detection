function lane_detection()

%% Clear everything.
clear all; close all;
%% Setup

% Generate clean up object
cleanupObj = onCleanup(@cleanMeUp);

% Read in video; capture initial frame for data.
vidObj = VideoReader('project_video.mp4');
img = readFrame(vidObj);
[h, w, d] = size(img);

% Points of ROI
pt1_x = w*0.22;
pt1_y = h*0.90;
pt2_x = w*0.93;
pt2_y = h*0.90;
pt3_x = w*0.59;
pt3_y = h*0.66;
pt4_x = w*0.42;
pt4_y = h*0.66;
roi_pts = [pt1_x pt1_y;
           pt2_x pt2_y;
           pt3_x pt3_y;
           pt4_x pt4_y;
           pt1_x pt1_y];

mask = poly2mask(roi_pts(:,1), roi_pts(:,2), h, w);
target_pts = [0 768; 202 768; 202 0; 0 0];

% Thresholding to this level will ignore sudden light changes.
level = 0.37;

% Unwarping
% fitgeotrans generates homography matrix in projective space
tf1 = fitgeotrans(roi_pts(1:4,:), target_pts, 'projective');
inv_tf1 = inv(tf1.T);
tf2 = fitgeotrans(target_pts, roi_pts(1:4,:), 'projective');
[img_tfm, b] = imwarp(img, tf1, 'OutputView',imref2d([768 202]));

% Define edges to discretize (group) lanes as left or right.
edges = 0:size(img_tfm, 2)/2:size(img_tfm);
h_tfm = size(img_tfm, 1);
w_tfm = size(img_tfm, 2);
slide_box_y = round(h_tfm/20:h_tfm/20:h_tfm-h_tfm/20);

% Circular Buffers
nbuff = 10;
seedBuffer = nan(2, nbuff);
LptsBuffer = nan(2,nbuff);
RptsBuffer = nan(2,nbuff);

% Text box coordinates
% txt_coords = [61 74; 160 74];
i = 0;
while hasFrame(vidObj)
    %% Unwarp the image for perspective.
    img = readFrame(vidObj);
    [img_tfm, b] = imwarp(img, tf1, 'OutputView',imref2d([768 202]));
    [img_uw, c] = imwarp(img_tfm, tf2, 'OutputView',imref2d([h w]));
    
    %% Processing
%     hsl_img = rgb2hsl(img_tfm);
%     sat_img = hsl_img(:,:,2);
    sat_img = img_tfm(:,:,1);
    sat_img = im2double(sat_img);
    imgA = imadjust(sat_img);
    imgM = medfilt2(imgA);
    imgG = imgaussfilt(imgM, 2);
    imgQ = imquantize(imgG, level);
    
    % Edge detection
    imgE = edge(imgG, 'Canny', [0.2 0.5]);
    
    
%     f = imshow(img_tfm);
%     set(f, 'Cdata', img_tfm)
%     hold on
    %% Sliding window centroid
    y_sum = sum(imgQ);
    [peaks, loc] = findpeaks(y_sum);
    groups = discretize(loc, edges);
    group1_loc = find(groups==1);
    group2_loc = find(groups==2);
    left_seed = round(mean(loc(group1_loc)));    
    right_seed = round(mean(loc(group2_loc)));
    
    % Circular buffer (if seed=NaN, use last known good value)
    if isnan(right_seed)
        right_seed = seedBuffer(1,1);
    elseif isnan(left_seed)
        right_seed = seedBuffer(2,1);
    elseif ~isnan(right_seed)
        seedBuffer = [right_seed seedBuffer(1,1:end-1); seedBuffer(2,:)];
    elseif ~isnan(left_seed)
        seedBuffer = [seedBuffer(1,:); left_seed seedBuffer(2,1:end-1)];
    end
    L_pts = [];
    R_pts = [];
    Y_pts = [];
    for kk = 1:size(slide_box_y, 2)   
            left_box_coords = [left_seed-30 slide_box_y(kk)-slide_box_y(1) 50 slide_box_y(1)*2];
            right_box_coords = [right_seed-30 slide_box_y(kk)-slide_box_y(1) 50 slide_box_y(1)*2];
            % UNCOMMENT to add sliding windows to transformed image.
%             r1 = rectangle('Position', left_box_coords);
%             r2 = rectangle('Position', right_box_coords);
%             r1.EdgeColor = 'b';
%             r2.EdgeColor = 'g';
            
            left_img_crop = imcrop(imgQ, left_box_coords);
            right_img_crop = imcrop(imgQ, right_box_coords);
            y_sumL = sum(left_img_crop);
            y_sumR = sum(right_img_crop);
            [Lpeaks, Lxloc] = findpeaks(y_sumL);
            [Rpeaks, Rxloc] = findpeaks(y_sumR);
            Rxloc = Rxloc+right_seed-30;
            yloc = round(slide_box_y(kk));
            
            % Plot Left coordinates
            try
%                 Lxloc2 = transformPointsForward(tf2, [Lxloc yloc]);
%                 plot(Lxloc2(1),Lxloc2(2), '*b')
%                 L_pts(1,kk) = Lxloc2(1);
%                 L_pts(2,kk) = Lxloc2(2);
                L_pts(1,kk) = Lxloc;
                L_pts(2,kk) = yloc;
%                 plot(Lxloc,yloc, '*g')
            catch
            end
            
            % Plot Right coordinates
            try
%                 Rxloc2 = transformPointsForward(tf2, [Rxloc yloc]);
%                 plot(Rxloc2(1),Rxloc2(2), '*g')
%                 R_pts(1,kk) = Rxloc2(1);
%                 R_pts(2,kk) = Rxloc2(2);
                R_pts(1,kk) = Rxloc;
                R_pts(2,kk) = yloc;
%                 plot(Rxloc,yloc, '*r')
            catch
            end
    end
    
    %% Plotting Lane detection
    
    % Remove all zeros from incoming arrays. TODO: why zeros?
    L_pts = L_pts(:,any(L_pts));
    R_pts = R_pts(:,any(R_pts));
    
    % Generate eqns of lines; use y->x to fit from (1,h_tfm)
    try
        P_left = polyfit(L_pts(2,:),L_pts(1,:), 1);
        LptsBuffer = [P_left(1) LptsBuffer(1,1:end-1);
                      P_left(2) LptsBuffer(2,1:end-1)];
    catch
        disp('Failed Left')
        P_left = LptsBuffer(:,1)';
    end
    try
        P_right = polyfit(R_pts(2,:),R_pts(1,:), 1);
        RptsBuffer = [P_right(1) RptsBuffer(1,1:end-1);
                      P_right(2) RptsBuffer(2,1:end-1)];
    catch
        disp('Failed Right')
        P_right = RptsBuffer(:,1)';
    end
    
    % Use polyval for interpolation from (1, h_tfm)
    Y = 1:1:h_tfm;
    XL = polyval(P_left, Y);
    XR = polyval(P_right, Y);
%     plot(XL, Y, 'r')
%     plot(XR, Y, 'r')

    % Unwarped: plot lines on both lanes.
    [XLT, YLT] = transformPointsForward(tf2, XL, Y);
    [XRT, YRT] = transformPointsForward(tf2, XR, Y);
    
    m_pts = [XLT(end) YLT(end);
            XRT(end) YRT(end);
            XRT(1) YRT(1);
            XLT(1) YLT(1)];
    
    unwarped_mask = poly2mask(m_pts(:,1), m_pts(:,2), h, w);
    
    img(unwarped_mask~=0)=0;
    f = imshow(img);
    set(f, 'Cdata', img)
    hold on
    
    % Plots transformed points
%     plot(L_pts(1,:), L_pts(2,:), '*g')
%     plot(R_pts(1,:), R_pts(2,:), '*r')

    
    plot(XLT, YLT, 'g')
    plot(XRT, YRT, 'r')
    
    % Curve display logic.
    if -0.0106 < P_left(1) && P_left(1) < 0.009
          text(61,100,'Bearing Left', 'FontSize',20, 'Color', 'g')
    elseif P_left(1) < -0.0106
          text(61,100,'Bearing Straight', 'FontSize',20, 'Color', 'b')
    end
        
    
    % Increase line width.
    set(findall(gca, 'Type', 'Line'),'LineWidth',5)
    

    % Quiver Plot overlay
%     mid_X = (XL + XR)/2;
%     mid_X = mid_X(:,1:90:end);
%     
%     mid_Y = linspace(1,h_tfm,10);
%     mid_Y = mid_Y(2:end);
%     
%     U = polyval(P_left, mid_Y);
%     U = U/(left_seed*100)*30;    
%     
%     V = linspace(-100,-100,9);
    
%     [XT, YT] = transformPointsForward(tf2, mid_X, mid_Y);
%     [UT, VT] = transformPointsForward(tf2, U, V);
%     q = quiver(XT, YT, U, V, 'r');
%     q.MaxHeadSize = 0.8;
%     Straight -0.0199   30.4374
    
    
    drawnow
%     export_fig(sprintf('D:/Users/alexg636/Dropbox/UMD Grad/4th Semester - 2018/ENPM673/Project/Project 1/images/LD_img%03d.jpg', i));
    hold off
    
    % Frame breaker at ~20 seconds
    if i == 525
       break 
    end
    i = i+1;
end
%% UNCOMMENT to write out the frames
% function cleanMeUp()
%     disp('Terminated')
%     % Image directory
%     workingDir = 'D:/Users/alexg636/Dropbox/UMD Grad/4th Semester - 2018/ENPM673/Project/Project 1/';
%     imgNames = dir(fullfile(workingDir, 'images', '*.jpg'));
%     imgNames = {imgNames.name};
%     
%     % Output Video
%     outputVid = VideoWriter(fullfile(workingDir, 'Lane_detection.avi'));
%     outputVid.FrameRate = 25;
%     open(outputVid)
%     
%     % Loop through images
%     for ii = 1:length(imgNames)
%        imgfile = imread(fullfile(workingDir, 'images', imgNames{ii}));
%        writeVideo(outputVid, imgfile)
%     end
%     
% end

end
