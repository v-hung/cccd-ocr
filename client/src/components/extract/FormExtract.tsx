import { useState, type FC, type HTMLAttributes } from "react";
import InputFloating from "../form/InputFloating";
import InputUpload from "../form/InputUpload";

type State = HTMLAttributes<HTMLDivElement> & {
	pageTitle: string;
	requestUrl: string;
};

type ExtractData = {
	extract: {
		id: string;
		name: string;
		dob: string;
		gender: string;
		origin_place: string;
	};
	content?: string;
};

const FormExtract: FC<State> = (props) => {
	const { className = "", pageTitle, requestUrl, ...rest } = props;

	const [data, setData] = useState<ExtractData | undefined>();
	const [loading, setLoading] = useState(false);

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
		try {
			if (loading) return;

			setLoading(true);

			e.preventDefault();

			const formData = new FormData(e.currentTarget);

			const data = await fetch(requestUrl, {
				method: "POST",
				body: formData,
			}).then((res) => res.json());

			setData(data);
		} catch (error) {
			alert("Something went wrong!");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div>
			<h3 className="text-2xl">{pageTitle}</h3>
			<form
				className="relative mt-4 flex max-w-5xl gap-4 rounded border border-gray-200 bg-white p-4 shadow"
				onSubmit={handleSubmit}
			>
				<div className="w-full lg:w-1/2">
					<InputUpload name="file" />
				</div>
				<div className="w-full lg:w-1/2">
					<div className="max-w-sm space-y-3">
						<InputFloating
							label="Id"
							value={data?.extract.id}
							readOnly={true}
						/>
						<InputFloating
							label="Name"
							value={data?.extract.name}
							readOnly={true}
						/>
						<InputFloating
							label="Date of birth"
							value={data?.extract.dob}
							readOnly={true}
						/>
						<InputFloating
							label="Gender"
							value={data?.extract.gender}
							readOnly={true}
						/>
						<InputFloating
							label="Origin place"
							value={data?.extract.origin_place}
							readOnly={true}
						/>
						<button
							type="submit"
							className="inline-flex items-center gap-x-2 rounded-lg border border-transparent bg-sky-600 px-4 py-3 text-sm font-medium text-white hover:bg-sky-700 focus:bg-sky-700 focus:outline-hidden disabled:pointer-events-none disabled:opacity-50"
						>
							Extract
						</button>
					</div>
				</div>

				{loading ? (
					<div className="absolute top-0 left-0 flex h-full w-full items-center justify-center bg-white/70">
						<IBxLoaderAlt className="size-11 animate-spin" />
					</div>
				) : null}
			</form>

			{data?.content ? (
				<div
					className="prose mt-6 max-w-full rounded border border-gray-200 bg-white p-4 shadow"
					dangerouslySetInnerHTML={{ __html: data.content }}
				></div>
			) : null}
		</div>
	);
};

export default FormExtract;
