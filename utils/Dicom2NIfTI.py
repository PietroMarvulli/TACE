import os
import itk


PixelType = itk.ctype("signed short")
PixelTypeUC = itk.ctype("unsigned char")
Dimension3D = 3
Dimension2D = 2
ImageType = itk.Image[PixelType, Dimension2D]
ImageTypeUC = itk.Image[PixelTypeUC, Dimension2D]
SerieType = itk.Image[PixelType, Dimension3D]
SerieTypeUC = itk.Image[PixelTypeUC, Dimension3D]

def orientedFilter(SerieReader, inType, outType):
    #print("\t\tApplying Oriented Filter...")
    ITK_COORDINATE_UNKNOWN = 0
    ITK_COORDINATE_Right = 2
    ITK_COORDINATE_Left = 3
    ITK_COORDINATE_Posterior = 4
    ITK_COORDINATE_Anterior = 5
    ITK_COORDINATE_Inferior = 8
    ITK_COORDINATE_Superior = 9
    ITK_COORDINATE_PrimaryMinor = 0
    ITK_COORDINATE_SecondaryMinor = 8
    ITK_COORDINATE_TertiaryMinor = 16

    ITK_COORDINATE_ORIENTATION_RAS = ((ITK_COORDINATE_Right) << (ITK_COORDINATE_PrimaryMinor)) \
                                     + ((ITK_COORDINATE_Anterior) << (ITK_COORDINATE_SecondaryMinor)) \
                                     + ((ITK_COORDINATE_Superior) << (ITK_COORDINATE_TertiaryMinor))

    orienterFilter = itk.OrientImageFilter[inType, outType].New()
    orienterFilter.UseImageDirectionOn()
    orienterFilter.SetDesiredCoordinateOrientation(ITK_COORDINATE_ORIENTATION_RAS)
    orienterFilter.SetInput(SerieReader.GetOutput())
    orienterFilter.Update()
    return orienterFilter.GetOutput()
def writeImages(image, imType, outputDir):
    writer = itk.ImageFileWriter[imType].New()
    outFileName = os.path.join(outputDir + ".nii.gz")
    writer.SetFileName(outFileName)
    writer.UseCompressionOn()
    writer.SetInput(image)
    writer.Update()
    return outFileName
def maskDicomProcessing(maskPath, outputDir):
    #print("\tMask Processing")
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(maskPath)

    seriesUID = namesGenerator.GetSeriesUIDs()

    for uid in seriesUID:
        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)
        reader = itk.ImageSeriesReader[SerieType].New()
        dicomIO = itk.GDCMImageIO.New()
        dicomIO.LoadPrivateTagsOn()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.Update()
        mask_path = writeImages(reader.GetOutput(), SerieType, outputDir)

    return mask_path
def seriesDicomProcessing(seriesPath, outputDir):
    #print("\tDICOM Serie Processing")
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(seriesPath)
    seriesUID = namesGenerator.GetSeriesUIDs()
    for uid in seriesUID:
        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)
        reader = itk.ImageSeriesReader[SerieType].New()
        dicomIO = itk.GDCMImageIO.New()
        dicomIO.LoadPrivateTagsOn()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.Update()
        image_path = writeImages(reader.GetOutput(), SerieType, outputDir)

def main():
    series_path = r"D:\dataset_TACE\HCC-TACE-Seg"
    new_dataset = r"D:\data_TACE\HCC-TACE-Seg"

    for patient in os.listdir(series_path)[2:3]:
        print("Conversion Series of Patient ",patient," ...")
        if not os.path.exists(os.path.join(new_dataset,patient)):
            os.mkdir(os.path.join(new_dataset,patient))
        for session in os.listdir(os.path.join(series_path, patient)):
            if not os.path.exists(os.path.join(new_dataset, patient, session)):
                os.mkdir(os.path.join(new_dataset, patient, session).replace(" ","_"))
            for series in os.listdir(os.path.join(series_path, patient, session)):
                if series.find("Segmentation") != -1:
                    maskDicomProcessing(os.path.join(series_path, patient, session, series),
                                        os.path.join(new_dataset, patient, session, series).replace(" ","_"))
                else:
                    seriesDicomProcessing(os.path.join(series_path, patient, session, series),
                                          os.path.join(new_dataset, patient, session, series).replace(" ","_"))
        print("\tConversion Finished! ")







if __name__ == '__main__':
    main()
