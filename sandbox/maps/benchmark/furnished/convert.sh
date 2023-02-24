#!/bin/bash

for d in */ ; do
    echo "$d"
    cd $d
    convert input.png -resize 80% -colorspace Gray resized.png
    convert resized.png -channel RGB -negate negated.png

    cornercolor=`convert negated.png -gravity southeast -format "%[pixel:u.p{0,0}]" info:`
    
    convert negated.png -background "$cornercolor" -gravity center -extent 2000x2000 extended.png
    
    convert extended.png -fill "gray(127)" -gravity southeast -draw "color 0,0 floodfill" rank.png

    cornercolor_check=`convert rank.png -gravity southeast -format "%[pixel:u.p{0,0}]" info:`
    
    rm resized.png negated.png extended.png
    cd ..
done
