import argparse
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299, help='image size to use')
    parser.add_argument('--vocabs_path', type=str, default='data/vocabs.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='coco' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--tb_log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    parser.add_argument('--val_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of gru hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in gru')
    parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
    parser.add_argument('--netM', type=str)
    parser.add_argument('--netR', type=str)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--val_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--kick', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--split', type=int, default=1)

    args = parser.parse_args()

    return args