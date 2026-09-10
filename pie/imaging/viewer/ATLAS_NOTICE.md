# Anatomical label lookup

`atlas.tsv` contains DKT/aseg label identifiers, names and RGB values drawn from
the existing local FastSurfer checkout's
`FastSurferCNN/config/FastSurfer_ColorLUT.tsv` and
`FastSurferCNN/config/FreeSurferColorLUT.txt`. The compact FastSurfer table omits
some right-hemisphere labels that are restored in the final segmentation; those
names/colors are included here from the full FreeSurfer lookup.

Sources: https://github.com/Deep-MI/FastSurfer and
https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles.
FastSurfer is licensed under Apache-2.0. This file is a lookup table, not an atlas
volume, registered template or synthetic anatomy. All displayed segmentations
come from the participant's local image-processing outputs.

DKT labels describe macroscopic anatomical parcels. They do not provide
individual functional boundaries, cellular anatomy, substantia nigra subnuclei,
or a clinical diagnosis. Custom atlases require their own explicit label lookup.
