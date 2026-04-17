# Pipeline
input: tgz file or list of file
output: restituted images, additional data, qc images and qc report

## 1. extract archive (cond)
input: tgz file
output: list of extracted files
desc: can be skipped if input is already a list of files

## 2. join images
input: list of tif files
output: mosaiced_image [TEMP]
desc: sub-pipeline
### 2.1 compute alignments
input: list of tif files
output: alignments (json)
### 2.2 mosaic images
input: list of tif files + alignments
output: mosaiced_image

## 3. generate quickview of mosaiced image (opt)
input: mosaiced_image
output: qv of mosaic image

## 4. restitute
input: mosaiced_image
output: final_image + qc
desc: sub-pipeline
### 4.1 detect vertical edges
input: mosaiced_image
output: vertical edges data + qc
### 4.2 detect horizontal edges
input: mosaiced_image + vertical edges data
output: horizontal edges data + qc
desc: tries collimation first, falls back to poly, then flat
### 4.3 apply transform
input: mosaiced_image + vertical edges data + horizontal edges data
output: final_image + qc
desc: selects TPS or Affine depending on which strategy succeeded in 4.2

## 5. generate quickview of final image (opt)
input: final_image
output: qv of final_image

## 6. generate qc report (opt)
input: all qc data
output: pdf report

## 7. clean temp files (opt)
input: all temp files (mosaiced_image, alignments, intermediate data)
output:
