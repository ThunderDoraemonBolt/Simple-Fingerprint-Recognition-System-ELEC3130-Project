clc; clear;

% change the folder path below to the folder storing the images
folderPath = 'C:\Users\path';
fileList = dir(fullfile(folderPath, '*.jpg'));

% create cells for all variables
cc = cell(1, numel(fileList));
images = cc;
eq_img = cc;
enhanced_img = cc;
binarized_img = cc;
thinned_img = cc;  % Added for thinned images
block_direction = cc;
roiImg = cc;
roiBound = cc;
roiArea = cc;
edgeDistance = cc;
end_list = cc;
branch_list = cc;
ridgeOrderMap = cc;
final_end = cc;
final_branch = cc;
pathMap = cc;
alignedRidgePoints = cc;
alignedAllPoints = cc;

% Processing parameters
blocksize = 16; % Standard block size for fingerprint processing
window_size = 5; % For coherence calculation
f = 0.5; % Enhancement parameter

% iteration for each image
for i = 1:numel(fileList)
    % Load and preprocess image
    filePath = fullfile(folderPath, fileList(i).name);
    images{i} = imread(filePath);
    if size(images{i}, 3) == 3
        images{i} = rgb2gray(images{i});
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.1 - Image Enhancement (inline version)
    I = 255-double(images{i});
    [w,h] = size(I);
    w1 = floor(w/32)*32;
    h1 = floor(h/32)*32;
    inner = zeros(w1,h1);
    
    for ii=1:32:w1
        for jj=1:32:h1
            a = ii+31;
            b = jj+31;
            F = fft2(I(ii:a,jj:b));
            factor = abs(F).^f;
            block = abs(ifft2(F.*factor));
            larv = max(block(:));
            if larv == 0
                larv = 1;
            end
            block = block./larv;
            inner(ii:a,jj:b) = block;
        end
    end
    
    enhanced_img{i} = histeq(uint8(inner*255));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.2 - Image Binarization (inline version)
    a = enhanced_img{i};
    [w_bin, h_bin] = size(a);
    o = zeros(w_bin, h_bin);
    W_bin = blocksize;
    
    for ii=1:W_bin:w_bin
        for jj=1:W_bin:h_bin
            if ii+W_bin-1 <= w_bin && jj+W_bin-1 <= h_bin
                mean_thres = mean2(a(ii:ii+W_bin-1,jj:jj+W_bin-1));
                mean_thres = 0.8*mean_thres;
                o(ii:ii+W_bin-1,jj:jj+W_bin-1) = a(ii:ii+W_bin-1,jj:jj+W_bin-1) < mean_thres;
            end
        end
    end
    
    binarized_img{i} = o;
    
    % Added Ridge Thinning
    thinned_img{i} = bwmorph(binarized_img{i}, 'thin', Inf);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.3 - Block Direction Estimation (inline version)
    image = binarized_img{i};
    [w_dir, h_dir] = size(image);
    direct = zeros(w_dir, h_dir);
    gradient_times_value = zeros(w_dir, h_dir);
    gradient_sq_minus_value = zeros(w_dir, h_dir);
    gradient_for_bg_under = zeros(w_dir, h_dir);
    W_dir = blocksize;
    blockIndex = zeros(ceil(w_dir/W_dir), ceil(h_dir/W_dir));
    center = [];
    
    filter_gradient = fspecial('sobel');
    I_horizontal = filter2(filter_gradient, image);
    filter_gradient = transpose(filter_gradient);
    I_vertical = filter2(filter_gradient, image);
    
    gradient_times_value = I_horizontal.*I_vertical;
    gradient_sq_minus_value = (I_vertical-I_horizontal).*(I_vertical+I_horizontal);
    gradient_for_bg_under = (I_horizontal.*I_horizontal) + (I_vertical.*I_vertical);
    
    for ii=1:W_dir:w_dir
        for jj=1:W_dir:h_dir
            if jj+W_dir-1 < h_dir && ii+W_dir-1 < w_dir
                times_value = sum(sum(gradient_times_value(ii:ii+W_dir-1, jj:jj+W_dir-1)));
                minus_value = sum(sum(gradient_sq_minus_value(ii:ii+W_dir-1, jj:jj+W_dir-1)));
                sum_value = sum(sum(gradient_for_bg_under(ii:ii+W_dir-1, jj:jj+W_dir-1)));
                
                if sum_value ~= 0 && times_value ~= 0
                    bg_certainty = (times_value*times_value + minus_value*minus_value)/(W_dir*W_dir*sum_value);
                    if bg_certainty > 0.05
                        blockIndex(ceil(ii/W_dir), ceil(jj/W_dir)) = 1;
                        tan_value = atan2(2*times_value, minus_value);
                        theta = (tan_value)/2;
                        theta = theta + pi/2;
                        center = [center; [round(ii + (W_dir-1)/2), round(jj + (W_dir-1)/2), theta]];
                    end
                end
            end
        end
    end
    
    x = bwlabel(blockIndex, 4);
    y = bwmorph(x, 'close');
    z = bwmorph(y, 'open');
    p = bwperim(z);
    
    block_direction{i} = z;
    direction_center{i} = center;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.4 - ROI Extraction (inline version)
    in = enhanced_img{i};
    inBound = p;
    inArea = z;
    
    [iw,ih] = size(in);
    tmplate = zeros(iw,ih);
    [w_area,h_area] = size(inArea);
    tmp = zeros(iw,ih);
    
    le2ri = sum(inBound);
    roiColumn = find(le2ri>0);
    left = min(roiColumn);
    right = max(roiColumn);
    
    tr_bound = inBound';
    up2dw = sum(tr_bound);
    roiRow = find(up2dw>0);
    upper = min(roiRow);
    bottom = max(roiRow);
    
    for ii = upper:1:bottom
        for jj = left:1:right
            if inBound(ii,jj) == 1
                tmplate(16*ii-15:16*ii,16*jj-15:16*jj) = 200;
                tmp(16*ii-15:16*ii,16*jj-15:16*jj) = 1;
            elseif inArea(ii,jj) == 1 && inBound(ii,jj) ~=1
                tmplate(16*ii-15:16*ii,16*jj-15:16*jj) = 100;
                tmp(16*ii-15:16*ii,16*jj-15:16*jj) = 1;
            end
        end
    end
    
    % Fix the data type mismatch
    if isinteger(in)
        tmp = im2uint8(tmp);
    else
        tmp = double(tmp);
    end
    in = in .* tmp;

    roiImg{i} = in(16*upper-15:16*bottom,16*left-15:16*right);
    roiBound{i} = inBound(upper:bottom,left:right);
    roiArea{i} = inArea(upper:bottom,left:right);
    roiArea{i} = im2double(roiArea{i}) - im2double(roiBound{i});
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.5 - Ridge Thinning (inline version)
    image_thin = thinned_img{i};
    inROI = roiArea{i};
    
    [w_thin,h_thin] = size(image_thin);
    a = sum(inROI);
    b = find(a>0);
    c = min(b);
    d = max(b);
    
    ii = round(w_thin/5);
    m = 0;
    for k=1:4
        m = m + sum(image_thin(k*ii,16*c:16*d));
    end
    e = (64*(d-c))/m;
    
    a = sum(inROI,2);
    b = find(a>0);
    c = min(b);
    d = max(b);
    ii = round(h_thin/5);
    m = 0;
    for k=1:4
        m = m + sum(image_thin(16*c:16*d,k*ii));
    end
    m = (64*(d-c))/m;
    
    edgeDistance{i} = round((m+e)/2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program no.6 - Combined Minutia Marking with Connection Prevention
in = thinned_img{i};
inimg = roiImg{i};
inBound = roiBound{i};
inArea = roiArea{i};
block = blocksize;

[w_mark,h_mark] = size(inimg);
[ridgeOrderMap{i},totalRidgeNum] = bwlabel(in);
edgeWidth = edgeDistance{i};

% Initialize lists
final_minutiae = [];  % Will store [x,y,type] where type: 0=end, 1=branch

for n=1:totalRidgeNum
    [m,n_coords] = find(ridgeOrderMap{i}==n);
    b = [m,n_coords];
    ridgeW = size(b,1);
    
    for x = 1:ridgeW
        ii_mark = b(x,1);
        jj_mark = b(x,2);
        
        % Calculate block indices with boundary checking
        block_i = ceil(ii_mark/block);
        block_j = ceil(jj_mark/block);
        
        % Only proceed if indices are within bounds
        if block_i >= 1 && block_i <= size(inArea,1) && ...
           block_j >= 1 && block_j <= size(inArea,2)
            
            if inArea(block_i,block_j) == 1
                % Ensure we don't access out of image bounds
                min_i = max(ii_mark-1, 1);
                max_i = min(ii_mark+1, size(in,1));
                min_j = max(jj_mark-1, 1);
                max_j = min(jj_mark+1, size(in,2));
                
                window = in(min_i:max_i,min_j:max_j);
                center_val = window(2,2);
                window(2,2) = 0; % Exclude center pixel
                neiborNum = sum(window(:));
                
                % For endpoints (neighbor count = 1)
                if neiborNum == 1
                    new_minutiae = [ii_mark, jj_mark, 0];
                    too_close = false;
                    % Check against all existing minutiae
                    for k = 1:size(final_minutiae,1)
                        dist = norm(new_minutiae(1:2) - final_minutiae(k,1:2));
                        if dist < edgeWidth/2
                            too_close = true;
                            break;
                        end
                    end
                    if ~too_close
                        final_minutiae = [final_minutiae; new_minutiae];
                    end
                    
                % For branches (neighbor count = 3)
                elseif neiborNum == 3
                    [br_y, br_x] = find(window==1);
                    branch_connected = false;
                    
                    % Check if any branch arm connects to existing minutiae
                    for arm = 1:3
                        global_pos = [ii_mark+br_y(arm)-2, jj_mark+br_x(arm)-2];
                        
                        % Check against all existing minutiae
                        for k = 1:size(final_minutiae,1)
                            dist = norm(global_pos - final_minutiae(k,1:2));
                            if dist < edgeWidth/2
                                branch_connected = true;
                                break;
                            end
                        end
                        if branch_connected, break; end
                    end
                    
                    if ~branch_connected
                        new_minutiae = [ii_mark, jj_mark, 1];
                        final_minutiae = [final_minutiae; new_minutiae];
                    end
                end
            end
        end
    end
end

% Final separation check (in case any were missed)
removed = false(size(final_minutiae,1),1);
for k = 1:size(final_minutiae,1)-1
    if removed(k), continue; end
    
    for m = k+1:size(final_minutiae,1)
        if removed(m), continue; end
        
        dist = norm(final_minutiae(k,1:2) - final_minutiae(m,1:2));
        if dist < edgeWidth/2
            % Prefer to keep endpoints over branches
            if final_minutiae(k,3) == 0 && final_minutiae(m,3) == 1
                removed(m) = true; % Remove branch
            elseif final_minutiae(k,3) == 1 && final_minutiae(m,3) == 0
                removed(k) = true; % Remove branch
                break;
            else % Same type, remove the second one
                removed(m) = true;
            end
        end
    end
end
final_minutiae = final_minutiae(~removed,:);

% Separate the final lists
end_list{i} = final_minutiae(final_minutiae(:,3)==0, 1:2);
branch_list{i} = final_minutiae(final_minutiae(:,3)==1, 1:2);
  % Create ridgeMap structure for alignment
    ridgeMap{i} = [];
    for n=1:totalRidgeNum
        [m,n_coords] = find(ridgeOrderMap{i}==n);
        ridgeMap{i} = [ridgeMap{i}; [m, n_coords, repmat(n,length(m),1)]];
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.7 - False Minutia Removal (inline version)
    in = roiImg{i};
    end_list{i}(:,3) = 0;
    branch_list{i}(:,3) = 1;
    minutiaeList = [end_list{i}; branch_list{i}];
    finalList = minutiaeList;
    [numberOfMinutia,~] = size(minutiaeList);
    suspectMinList = [];
    
    for ii=1:numberOfMinutia-1
        for jj=ii+1:numberOfMinutia
            d = sqrt((minutiaeList(ii,1)-minutiaeList(jj,1))^2 + (minutiaeList(ii,2)-minutiaeList(jj,2))^2);
            if d < edgeWidth
                suspectMinList = [suspectMinList; [ii,jj]];
            end
        end
    end
    
    [totalSuspectMin,~] = size(suspectMinList);
    for k=1:totalSuspectMin
        typesum = minutiaeList(suspectMinList(k,1),3) + minutiaeList(suspectMinList(k,2),3);
        
        if typesum == 1 || typesum == 2
            if ridgeOrderMap{i}(minutiaeList(suspectMinList(k,1),1), minutiaeList(suspectMinList(k,1),2)) == ...
               ridgeOrderMap{i}(minutiaeList(suspectMinList(k,2),1), minutiaeList(suspectMinList(k,2),2))
                finalList(suspectMinList(k,1),1:2) = [-1,-1];
                finalList(suspectMinList(k,2),1:2) = [-1,-1];
            end
        elseif typesum == 0
            a = minutiaeList(suspectMinList(k,1),1:3);
            b = minutiaeList(suspectMinList(k,2),1:3);
            
            if ridgeOrderMap{i}(a(1),a(2)) ~= ridgeOrderMap{i}(b(1),b(2))
                finalList(suspectMinList(k,1),1:2) = [-1,-1];
                finalList(suspectMinList(k,2),1:2) = [-1,-1];
            else
                finalList(suspectMinList(k,1),1:2) = [-1,-1];
                finalList(suspectMinList(k,2),1:2) = [-1,-1];
            end
        end
    end
    
    final_end{i} = [];
    final_branch{i} = [];
    pathMap{i} = [];
    
    for k=1:numberOfMinutia
        if finalList(k,1:2) ~= [-1,-1]
            if finalList(k,3) == 0
                final_end{i} = [final_end{i}; [finalList(k,1:2), 0]]; % 0 is placeholder for theta
            else
                final_branch{i} = [final_branch{i}; finalList(k,1:2)];
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Program no.8 - Alignment Stage (inline version)
    if ~isempty(final_end{i}) && ~isempty(ridgeMap{i})
        real_end = final_end{i};
        k = 1; % Using first minutiae point as reference
        
        % First alignment function (ridge points)
        theta = real_end(k,3);
        if theta < 0
            theta1 = 2*pi + theta;
        else
            theta1 = theta;
        end
        theta1 = pi/2 - theta1;
        rotate_mat = [cos(theta1), -sin(theta1); sin(theta1), cos(theta1)];
        
        pathPointForK = find(ridgeMap{i}(:,3) == k);
        if ~isempty(pathPointForK)
            toBeTransformedPointSet = ridgeMap{i}(min(pathPointForK):max(pathPointForK),1:2)';
            tonyTrickLength = size(toBeTransformedPointSet,2);
            pathStart = real_end(k,1:2)';
            translatedPointSet = toBeTransformedPointSet - pathStart(:,ones(1,tonyTrickLength));
            alignedRidgePoints{i} = rotate_mat * translatedPointSet;
        end
        
        % Second alignment function (all points)
        theta_all = real_end(k,3);
        if theta_all < 0
            theta1_all = 2*pi + theta_all;
        else
            theta1_all = theta_all;
        end
        theta1_all = pi/2 - theta1_all;
        rotate_mat_all = [cos(theta1_all), -sin(theta1_all), 0; 
                         sin(theta1_all), cos(theta1_all), 0; 
                         0, 0, 1];
        
        toBeTransformedPointSet_all = real_end';
        tonyTrickLength_all = size(toBeTransformedPointSet_all,2);
        pathStart_all = real_end(k,:)';
        translatedPointSet_all = toBeTransformedPointSet_all - pathStart_all(:,ones(1,tonyTrickLength_all));
        newXY_all = rotate_mat_all * translatedPointSet_all;
        
        for m = 1:tonyTrickLength_all
            if or(newXY_all(3,m) > pi, newXY_all(3,m) < -pi)
                newXY_all(3,m) = 2*pi - sign(newXY_all(3,m)) * newXY_all(3,m);
            end
        end
        alignedAllPoints{i} = newXY_all;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enhanced Visualization for each image
figure('Name', ['Fingerprint Analysis: ' fileList(i).name], 'Position', [100, 100, 1400, 900]);

% Original processing pipeline
subplot(3,5,1); imshow(images{i}); title('Original');
subplot(3,5,2); imshow(enhanced_img{i}); title('Enhanced');
subplot(3,5,3); imshow(binarized_img{i}); title('Binarized');
subplot(3,5,4); imshow(thinned_img{i}); title('Thinned');

% ROI visualizations
subplot(3,5,5); imshow(roiImg{i}); title('ROI');
subplot(3,5,6); imshow(roiBound{i}); title('ROI Boundary');
subplot(3,5,7); imshow(roiArea{i}); title('ROI Area');

% Minutiae on original ROI with better visualization
subplot(3,5,8); 
imshow(roiImg{i});
hold on;
if ~isempty(end_list{i})
    plot(end_list{i}(:,2), end_list{i}(:,1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
if ~isempty(branch_list{i})
    plot(branch_list{i}(:,2), branch_list{i}(:,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title('Minutiae Detection');

% Minutiae on thinned ROI
subplot(3,5,9);
thinned_roi = thinned_img{i}(16*upper-15:16*bottom,16*left-15:16*right);
imshow(thinned_roi);
hold on;
if ~isempty(end_list{i})
    plot(end_list{i}(:,2), end_list{i}(:,1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
if ~isempty(branch_list{i})
    plot(branch_list{i}(:,2), branch_list{i}(:,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title('Minutiae on Thinned ROI');

% False minutiae removal on thinned
subplot(3,5,10);
imshow(thinned_roi);
hold on;
if ~isempty(final_end{i})
    plot(final_end{i}(:,2), final_end{i}(:,1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
if ~isempty(final_branch{i})
    plot(final_branch{i}(:,2), final_branch{i}(:,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title('After False Minutiae Removal');

% Block direction visualization
subplot(3,5,11); 
imagesc(block_direction{i}); 
colormap(gray); 
hold on;
if ~isempty(direction_center{i})
    [u,v] = pol2cart(direction_center{i}(:,3),8);
    quiver(direction_center{i}(:,2),direction_center{i}(:,1),u,v,0,'g');
end
hold off;
title('Block Direction');

% Aligned fingerprint visualization
if ~isempty(alignedAllPoints{i})
    subplot(3,5,12);
    % Create a blank canvas for aligned print
    canvas_size = max(abs(alignedAllPoints{i}(1:2,:)),[],2)*2;
    canvas = zeros(round(canvas_size(2)), round(canvas_size(1)));
    imshow(canvas);
    hold on;
    
    % Plot aligned minutiae
    scatter(alignedAllPoints{i}(1,:)+canvas_size(1)/2, ...
            alignedAllPoints{i}(2,:)+canvas_size(2)/2, ...
            50, 'filled');
    
    % Plot direction vectors
    [u,v] = pol2cart(alignedAllPoints{i}(3,:),15);
    quiver(alignedAllPoints{i}(1,:)+canvas_size(1)/2,...
           alignedAllPoints{i}(2,:)+canvas_size(2)/2,...
           u, v, 0, 'r', 'LineWidth', 1.5);
    
    % Mark reference point
    plot(canvas_size(1)/2, canvas_size(2)/2, 'yx', 'MarkerSize', 15, 'LineWidth', 3);
    
    hold off;
    axis equal;
    title('Aligned Minutiae with Directions');
end

% Aligned ridge points visualization
if ~isempty(alignedRidgePoints{i})
    subplot(3,5,13);
    scatter(alignedRidgePoints{i}(1,:), alignedRidgePoints{i}(2,:), 5, 'filled');
    axis equal; grid on;
    title('Aligned Ridge Points');
end

% Final minutiae classification
subplot(3,5,14);
imshow(roiImg{i});
hold on;
if ~isempty(final_end{i})
    plot(final_end{i}(:,2), final_end{i}(:,1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(final_end{i}(:,2)+5, final_end{i}(:,1), 'E', 'Color', 'r', 'FontSize', 10);
end
if ~isempty(final_branch{i})
    plot(final_branch{i}(:,2), final_branch{i}(:,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
    text(final_branch{i}(:,2)+5, final_branch{i}(:,1), 'B', 'Color', 'g', 'FontSize', 10);
end
hold off;
title('Classified Minutiae (E=End, B=Branch)');

sgtitle(['Fingerprint Analysis: ' fileList(i).name], 'FontSize', 16, 'FontWeight', 'bold');
end