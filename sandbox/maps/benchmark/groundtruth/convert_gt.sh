#!/bin/bash

for d in */ ; do
    echo "$d"
    cd $d
    
    convert input.png -threshold 98% threshold.png
    
    convert threshold.png -resize 80% -colorspace Gray resized.png

    convert resized.png -threshold 98% binary.png

    # convert binary.png -channel RGB -negate negated.png

    cornercolor=`convert binary.png -gravity southeast -format "%[pixel:u.p{0,0}]" info:`
    
    convert binary.png -background "$cornercolor" -gravity center -extent 2000x2000 label.png
    
    rm threshold.png resized.png binary.png
    cd ..
done
