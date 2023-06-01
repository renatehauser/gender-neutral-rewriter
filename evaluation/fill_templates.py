import os
import argparse


def main(input_dir, output_dir, templates):
    # Find all bucket files in the input directory
    bucket_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Process each bucket file
    for bucket_file in bucket_files:
        # Open the input file for reading
        with open(os.path.join(input_dir, bucket_file), "r") as f:
            terms = [line.strip() for line in f.readlines()]

        filled_templates = []
        # Process each term and write the resulting segments to the output file
        for term in terms:
            # Generate the segments for this term by filling in the templates
            segments = [template.format(term=term) for template in templates]
            filled_templates.extend(segments)

        # Write the segments to the output file
        with open(os.path.join(output_dir, f"{bucket_file}"), "w") as f:
            for filled_seg in filled_templates:
                f.write(f"{filled_seg.strip()}\n")
            #f.writelines(filled_templates)
            #f.write("\n")


if __name__ == '__main__':
    # Define the available command-line arguments
    parser = argparse.ArgumentParser(description='Process bucket files and generate template segments.')
    parser.add_argument('--indir', help='the input directory containing the bucket files')
    parser.add_argument('--outdir', help='the output directory for the resulting segments')
    parser.add_argument('-t', '--templates', help='Path to file that contains list of templates to fill in')

    # Parse the command-line arguments
    args = parser.parse_args()

    with open(args.templates, 'r', encoding='utf-8') as t:
        templates = t.readlines()

    # Call the main function with the given arguments
    main(args.indir, args.outdir, templates)
