// adding noise profiles to the image

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h" 
#include "itkImageRegionIterator.h"
#include "itkMirrorPadImageFilter.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>

int main(int argc, char* argv[])
{
  // important parameters (user defined)
  int dir = 12;       // direction (between 0 and 64, 0 is for the b0 and the rest is for dwis)
  int s = 2;          // search radius
  int r = 3;          // number of q neighbors
  float beta = 0.01;  // beta value
  int M;              // X size
  int N;              // Y size
  int L;              // Z size
  
  // important parameters (computed)
  int W = 2*s+1;     // search diameter
  
  // define the pixel type and dimension
  typedef float PixelType;
  const unsigned int Dimension = 4; 

  // define the image types
  typedef itk::Image< PixelType, Dimension > InImageType;
  typedef itk::Image< PixelType, Dimension-1 > OutImageType;
  
  // define the reader and writer types
  typedef itk::ImageFileReader< InImageType > ReaderType;
  typedef itk::ImageFileWriter< OutImageType > WriterType;

  std::string inputVolumeFileName = argv[1];
  std::string sigmaVolumeFileName = argv[2];

  // read the input volume
  ReaderType::Pointer I_reader = ReaderType::New();
  I_reader->SetFileName(inputVolumeFileName);
  I_reader->Update();
  InImageType::Pointer I = I_reader->GetOutput();
  InImageType::RegionType im_region = I->GetLargestPossibleRegion();
  InImageType::SizeType im_size = im_region.GetSize();

  // M = im_region[0];
  // N = im_region[1];
  // L = im_region[2];

  std::cout << "Input Volume Size : " << im_size << std::endl; // print the volume size

  // pad the input volume spatially
  typedef itk::MirrorPadImageFilter< InImageType, InImageType > MirrorPadFilterType;

  InImageType::SizeType lowerBound;
  lowerBound[0] = s;
  lowerBound[1] = s;
  lowerBound[2] = s;
  lowerBound[3] = 0;

  InImageType::SizeType upperBound;
  upperBound[0] = s;
  upperBound[1] = s;
  upperBound[2] = s;
  upperBound[3] = 0;

  MirrorPadFilterType::Pointer mirror_pad_filter = MirrorPadFilterType::New();
  
  mirror_pad_filter->SetInput( I );
  mirror_pad_filter->SetPadLowerBound( lowerBound );
  mirror_pad_filter->SetPadUpperBound( upperBound );
  mirror_pad_filter -> Update();  

  InImageType::Pointer I_padded = mirror_pad_filter->GetOutput();
  
  im_region = I_padded->GetLargestPossibleRegion();
  im_size = im_region.GetSize();

  std::cout << "Input Volume Size (After Padding): " << im_size << std::endl; // print the volume size

  // read the sigma volume
  ReaderType::Pointer Sigma_reader = ReaderType::New();
  Sigma_reader->SetFileName(sigmaVolumeFileName);
  Sigma_reader->Update();
  InImageType::Pointer Sigma = Sigma_reader->GetOutput();
  InImageType::RegionType sig_region = Sigma->GetLargestPossibleRegion();
  InImageType::SizeType sig_size = sig_region.GetSize();

  std::cout << "Input Sigma Size : " << sig_size << std::endl; // print the volume size

  // extract the sigma volume for the direction "dir"
  typedef itk::ExtractImageFilter< InImageType, OutImageType > SigmaExtractFilterType;
  SigmaExtractFilterType::Pointer sigma_extract_volume = SigmaExtractFilterType::New();

  sigma_extract_volume->SetDirectionCollapseToSubmatrix();

  sig_size[3] = 0;

  InImageType::IndexType start = sig_region.GetIndex();
  start[3] = dir;

  InImageType::RegionType desiredSigmaRegion;
  desiredSigmaRegion.SetSize(  sig_size  );
  desiredSigmaRegion.SetIndex( start ); 
  
  sigma_extract_volume->SetExtractionRegion( desiredSigmaRegion );
  sigma_extract_volume->SetInput( Sigma );
  sigma_extract_volume->Update();
  OutImageType::Pointer Sigma_dir = sigma_extract_volume->GetOutput();

  // read the bvecs
  std::ifstream bvec("bvec.txt");
  std::vector<std::vector<float> > v(3, std::vector<float>(65, 0));
    
  for (int i = 0; i < v.size(); i++) {
      for (int j = 0; j < v[i].size(); j++){
          bvec >> v[i][j];
      }
  }

  // read the q-neighbors
  std::ifstream q("q_neighbors.txt");
  std::vector<std::vector<int> > q_neigbors(64, std::vector<int>(4, 0));
    
  for (int i = 0; i < q_neigbors.size(); i++) {
      for (int j = 0; j < q_neigbors[i].size(); j++){
          q >> q_neigbors[i][j];
      }
  }

  // iteration
  OutImageType::Pointer I_denoised = OutImageType::New();
  OutImageType::SizeType I_denoised_size = {{55, 55, 55}};
  OutImageType::IndexType I_denoised_start = {{0, 0, 0}};
  OutImageType::RegionType I_denoised_region;
  I_denoised_region.SetSize(I_denoised_size);
  I_denoised_region.SetIndex(I_denoised_start);
  I_denoised->SetRegions(I_denoised_region);
  I_denoised->Allocate();

  InImageType::SizeType I_size = {{55, 55, 55, 1}};
  InImageType::IndexType I_start = {{0, 0, 0, dir}};
  InImageType::RegionType I_region; 
  I_region.SetSize(I_size);
  I_region.SetIndex(I_start); 

  // typedef itk::ConstNeighborhoodIterator< InImageType > InNeighborhoodIteratorType; 
  typedef itk::ImageRegionIterator< InImageType > InIteratorType;
  typedef itk::ImageRegionIterator< OutImageType > OutIteratorType;
  
  InIteratorType in(I, I_region);
  OutIteratorType out(I_denoised, I_denoised->GetRequestedRegion());

  for (in.GoToBegin(), out.GoToBegin(); !in.IsAtEnd();++in, ++out){
    out.Set(in.Get());
  }
  
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("output.nii.gz");
  writer->SetInput(I_denoised);

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


