// adding noise profiles to the image

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <iostream>
#include <string>
#include <cmath>
#include "itkNormalVariateGenerator.h"
#include "itkStatisticsImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkMaskImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"

int main(int argc, char* argv[])
{
  typedef unsigned char PixelType;
  const unsigned int Dimension = 4; 

  typedef itk::Image< PixelType, Dimension > ImageType;

  typedef itk::ImageRegionConstIterator< ImageType > ConstIteratorType;
  typedef itk::ImageRegionIterator< ImageType>       IteratorType;

  typedef itk::ImageFileReader< ImageType > ReaderType;
  typedef itk::ImageFileWriter< ImageType > WriterType;

  typedef itk::Statistics::NormalVariateGenerator NormalGeneratorType;
  NormalGeneratorType::Pointer normalGenerator = NormalGeneratorType::New();

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  std::string inputFileName = argv[1];
  std::string outputFileName = argv[2];
  std::string maskFileName = argv[3];
  std::cout << inputFileName <<std::endl; 
  std::cout << outputFileName << std::endl;

  // get the image size
  reader->SetFileName(inputFileName);
  reader->Update();
  ImageType::Pointer image = reader->GetOutput();
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();

  std::cout << size << std::endl; // print the image size

  // get the maximum intensity of the image
  typedef itk::MinimumMaximumImageCalculator<ImageType> ImageCalculatorFilterType;

  ImageCalculatorFilterType::Pointer imageCalculatorFilter = ImageCalculatorFilterType::New();
  imageCalculatorFilter->SetImage(image);
  imageCalculatorFilter->ComputeMaximum();
  unsigned char sigma;
  sigma = imageCalculatorFilter->GetMaximum();
  sigma = sigma * 5 / 100;

  std::cout << sigma << std::endl; // print sigma

  // get the image mean
  // typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
  // StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New ();
  // statisticsImageFilter->SetInput(image);
  // statisticsImageFilter->Update();
  // unsigned char sigma;
  // sigma = statisticsImageFilter->GetMean();
  // sigma = sigma / 10;

  // add rician noise
  ImageType::RegionType inputRegion;
  ImageType::RegionType::IndexType inputStart;

  inputStart[0] = 0;
  inputStart[1] = 0;
  inputStart[2] = 0;
  inputStart[3] = 0;

  inputRegion.SetSize(size);
  inputRegion.SetIndex(inputStart);

  ImageType::RegionType outputRegion;
  ImageType::RegionType::IndexType outputStart;

  outputStart[0] = 0;
  outputStart[1] = 0;
  outputStart[2] = 0;
  outputStart[3] = 0;

  outputRegion.SetSize( size );
  outputRegion.SetIndex( outputStart );

  ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions(outputRegion);
  const ImageType::SpacingType& spacing = reader->GetOutput()->GetSpacing();
  const ImageType::PointType& inputOrigin = reader->GetOutput()->GetOrigin();
  double   outputOrigin[Dimension];

  for(unsigned int i=0; i< Dimension; i++)
  {
  outputOrigin[i] = inputOrigin[i] + spacing[i] * inputStart[i];
  }

  outputImage->SetSpacing( spacing );
  outputImage->SetOrigin(  outputOrigin );
  outputImage->Allocate();

  ConstIteratorType inputIt(   reader->GetOutput(), inputRegion  );
  IteratorType      outputIt(  outputImage,         outputRegion );

  inputIt.GoToBegin();
  outputIt.GoToBegin();

  while( !inputIt.IsAtEnd() )
    {
    unsigned char I;
    unsigned char epsilon1;
    unsigned char epsilon2;
    I = inputIt.Get();
    epsilon1 = ( normalGenerator->GetVariate() * sigma ) + 0;
    epsilon2 = ( normalGenerator->GetVariate() * sigma ) + 0;
    outputIt.Set(std::pow(std::pow(I + epsilon1,2) + epsilon2, 0.5));
    ++inputIt;
    ++outputIt;
    }

  // get masked images
  typedef itk::MaskImageFilter< ImageType, ImageType > MaskImageFilterType;

  MaskImageFilterType::Pointer MaskFilter = MaskImageFilterType::New();

  typedef unsigned char MaskPixelType;
  const unsigned int MaskDimension = 4; 

  typedef itk::Image< MaskPixelType, MaskDimension > MaskImageType;

  typedef itk::ImageFileReader< MaskImageType > MaskReaderType;
  ReaderType::Pointer mask_reader = ReaderType::New();

  mask_reader->SetFileName(maskFileName);
  mask_reader->Update();
  ImageType::Pointer mask_image = mask_reader->GetOutput();
  ImageType::RegionType mask_region = mask_image->GetLargestPossibleRegion();
  ImageType::SizeType mask_size = mask_region.GetSize();

  std::cout << mask_size << std::endl; // print the image size

  // resample the mask

  typedef itk::ResampleImageFilter<MaskImageType, MaskImageType> ResampleImageFilterType;
  ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();

  typedef itk::AffineTransform< double, Dimension > TransformType;
  TransformType::Pointer transform = TransformType::New();

  typedef double ScalarType;
  typedef itk::LinearInterpolateImageFunction<MaskImageType, ScalarType> LinearInterpolatorType;
  LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();

  TransformType::OutputVectorType translation1;

  const ImageType::PointType& ref_origin = outputImage->GetOrigin();
  const ImageType::SpacingType& ref_spacing = outputImage->GetSpacing();
  const MaskImageType::PointType& mask_origin = mask_image->GetOrigin();

  const double t1 = mask_origin[0] -  mask_origin[0];
  const double t2 = mask_origin[1] -  mask_origin[1];
  const double t3 = mask_origin[2] -  mask_origin[2];
  const double t4 = mask_origin[3] -  mask_origin[3];

  translation1[0] = t1;
  translation1[1] = t2;
  translation1[2] = t3;
  translation1[3] = t4;

  transform->Translate(translation1);

  // resampleFilter->SetReferenceImage(outputImage);
  resampleFilter->SetInput(mask_image);
  resampleFilter->SetSize(outputImage->GetLargestPossibleRegion().GetSize());
  resampleFilter->SetOutputOrigin( ref_origin );
  resampleFilter->SetOutputSpacing( ref_spacing );
  // resampleFilter->SetInterpolator(interpolator);
  // resampleFilter->UseReferenceImageOn();
  // resampleFilter->GetUseReferenceImage();
  resampleFilter->SetTransform(transform);
 
  MaskFilter->SetInput(outputImage);
  MaskFilter->SetMaskImage(resampleFilter->GetOutput());

  // write the image
  writer->SetFileName(outputFileName);
  writer->SetInput(resampleFilter->GetInput());

  try
  {
	writer->Update();
  }
  catch(itk::ExceptionObject & exp)
  {
	  std::cout << exp << std::endl;
  }
  
  return EXIT_SUCCESS;
}


