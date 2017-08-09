from pyrouge import Rouge155
import tempfile, os, glob, shutil

def evaluate_rouge(summaries, references, remove_temp=False, rouge_args=[]):
    temp_dir = tempfile.mkdtemp()
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print temp_dir, system_dir, model_dir

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'wb') as f:
                f.writelines([x + '\n' for x in candidate])

        with open(os.path.join(system_dir, summary_fn), 'wb') as f:
            f.writelines([x + '\n' for x in summary])

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'
    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    print output

    # remove the created temporary files
    if remove_temp:
        shutil.rmtree(temp_dir)
    return r

if __name__ == '__main__':
    references = []
    print glob.glob('/Users/dai/dev/ml/pyrouge/pyrouge/tests/data/models_plain/D30001.*')
    # load the sample references
    candidates = []
    for fn in glob.glob('/Users/dai/dev/ml/pyrouge/pyrouge/tests/data/models_plain/D30001.*'):
        with open(fn) as f:
            reference = [x.strip() for x in f.readlines()]
        candidates.append(reference)
    references.append(candidates)
    print len(references)

    summaries = []
    with open('/Users/dai/dev/ml/pyrouge/pyrouge/tests/data/systems_plain/D30001.M.100.T.A') as f:
        summary = [x.strip() for x in f.readlines()]
    summaries.append(summary)

    rouge_args = [
        '-c', 95,
        '-U',
        '-r', 1,
        '-n', 2,
        '-a',
    ]
    print evaluate_rouge(summaries, references, True, rouge_args)
